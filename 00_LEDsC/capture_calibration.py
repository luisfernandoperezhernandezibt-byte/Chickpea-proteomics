#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys
import time
import datetime
import statistics

try:
    import serial  # pyserial
except ImportError:
    sys.exit("ERROR: pyserial not installed.  Run:  pip install pyserial")

# --------------------------- USER SETTINGS ----------------------------------
PORT = "COM9"               # <-- CHANGE to your ESP32 port
BAUD = 115200
DATA_DIRNAME = "calibration_data"   # subfolder (created next to this script)
EXPORT_XLSX = True

DISTANCES_CM = [1, 3, 5, 8, 12, 18, 26, 35, 45]   # locked study sweep
REPLICATES = 4

COLOR_PRESETS = {
    "1": ("Red",   "660"),
    "2": ("Green", "535"),
    "3": ("Blue",  "450"),
}

# stability detection ("Normal" preset)
CV_THRESHOLD = 0.01     # accept when last readings vary < 1%
STAB_WINDOW  = 3        # number of recent readings compared
STAB_TIMEOUT = 25       # seconds before offering manual fallback
# ----------------------------------------------------------------------------

CH_NAMES = ["F1_415", "F2_445", "F3_480", "F4_515", "F5_555",
            "F6_590", "F7_630", "F8_680", "Clear", "NIR"]

SENSOR_COLS = (["gain", "atime", "astep", "integration_ms", "burst", "saturated"]
               + [f"mean_{c}" for c in CH_NAMES]
               + [f"sd_{c}" for c in CH_NAMES])

META_COLS = ["sample_id", "timestamp", "color", "nominal_wavelength_nm",
             "intensity", "reference_name", "reference_ppfd", "distance_cm",
             "replicate", "sensor_temp_note", "notes"]

HEADER = META_COLS + SENSOR_COLS
N_SENSOR_FIELDS = len(SENSOR_COLS)


def script_dir():
    """Folder where THIS .py file lives (independent of the terminal's folder)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def make_outfile():
    """Create calibration_data/ next to the script and return a timestamped path."""
    ddir = os.path.join(script_dir(), DATA_DIRNAME)
    os.makedirs(ddir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    path = os.path.join(ddir, f"calibration_{stamp}.csv")
    # if two runs start within the same minute, add seconds to stay unique
    if os.path.exists(path):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = os.path.join(ddir, f"calibration_{stamp}.csv")
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)
    return path


def connect():
    print(f"Opening {PORT} @ {BAUD} baud ...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=12)
    except serial.SerialException as e:
        sys.exit(f"ERROR: could not open {PORT}: {e}\n"
                 f"  - Check PORT matches Arduino IDE -> Tools -> Port.\n"
                 f"  - Make sure the Arduino Serial Monitor is CLOSED.")
    time.sleep(2.5)                      # ESP32 resets when port opens
    ser.reset_input_buffer()
    for _ in range(10):                  # robust handshake (fixes 'no PONG')
        ser.write(b"PING\n")
        line = ser.readline().decode(errors="ignore").strip()
        if line in ("PONG", "READY"):
            print("Sensor link OK.\n")
            return ser
        time.sleep(0.4)
    print("WARNING: no reply to PING; the sensor may still work. Continuing.\n")
    return ser


def read_once(ser, retries=2):
    """Return (means_list[10], saturated_bool, raw_fields_list) or (None, False, None)."""
    for _ in range(retries + 1):
        ser.reset_input_buffer()
        ser.write(b"READ\n")
        deadline = time.time() + 12
        while time.time() < deadline:
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("DATA"):
                parts = line.split(",")[1:]
                if len(parts) == N_SENSOR_FIELDS:
                    means = [float(x) for x in parts[6:16]]
                    saturated = parts[5] == "1"
                    return means, saturated, parts
                break
            if line.startswith("ERROR"):
                print("   sensor:", line)
                return None, False, None
    return None, False, None


def signal_of(means):
    """Quantity whose stability we track: sum of the 8 visible channels."""
    return sum(means[:8])


def wait_until_stable(ser):
    """
    Show live counts; return (parts, status) where status is:
      'stable'    -> variation < threshold (parts is a good reading)
      'timeout'   -> never settled (parts is the last reading)
      'saturated' -> a reading saturated (parts is None)
      'error'     -> read failure (parts is None)
    """
    history = []
    last_parts = None
    t0 = time.time()
    print("   waiting for a stable reading (variation < 1%) ...")
    while time.time() - t0 < STAB_TIMEOUT:
        means, saturated, parts = read_once(ser)
        if parts is None:
            return None, "error"
        if saturated:
            print("   *** SATURATED *** - add a PTFE layer or increase distance.")
            return None, "saturated"
        last_parts = parts

        sig = signal_of(means)
        history.append(sig)
        history = history[-STAB_WINDOW:]

        dom_i = max(range(8), key=lambda i: means[i])
        if len(history) >= STAB_WINDOW and statistics.mean(history) > 0:
            cv = statistics.pstdev(history) / statistics.mean(history)
            tail = f"   {CH_NAMES[dom_i]}: {means[dom_i]:8.0f}   variation={cv*100:4.1f}%  "
            if cv < CV_THRESHOLD:
                print(tail + "OK - stable")
                return parts, "stable"
            print(tail + "(settling)")
        else:
            print(f"   {CH_NAMES[dom_i]}: {means[dom_i]:8.0f}   (collecting)")
    return last_parts, "timeout"


def record_point(ser, outfile, sid, color, wl, intensity, ref_name, ref, d, temp_note):
    """Record REPLICATES rows in the exact META_COLS order. Returns (sid, n_saved)."""
    saved = 0
    for rep in range(1, REPLICATES + 1):
        means, saturated, parts = read_once(ser)
        if parts is None:
            print(f"      replicate {rep}: read failed - skipping")
            continue
        if saturated:
            print(f"      replicate {rep}: SATURATED - not saved")
            continue
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = ([sid, ts, color, wl, intensity, ref_name, ref, d, rep, temp_note, ""]
               + parts)
        with open(outfile, "a", newline="") as f:
            csv.writer(f).writerow(row)
        sid += 1
        saved += 1
        print(f"      replicate {rep}/{REPLICATES}  saved")
    return sid, saved


def ask(prompt, default=""):
    v = input(prompt).strip()
    return v if v else default


def export_xlsx(csv_path):
    try:
        import openpyxl
    except ImportError:
        print("(install openpyxl for .xlsx export:  pip install openpyxl)")
        return
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "calibration"
    with open(csv_path) as f:
        for row in csv.reader(f):
            out = []
            for v in row:
                try:
                    out.append(float(v) if ("." in v or v.lstrip("-").isdigit()) else v)
                except ValueError:
                    out.append(v)
            ws.append(out)
    ws.freeze_panes = "A2"
    xlsx = os.path.splitext(csv_path)[0] + ".xlsx"
    wb.save(xlsx); print(f"Also wrote {xlsx}")


def choose_color():
    """Show the colour menu; return (name, wavelength) or None to finish."""
    print("\n" + "=" * 64)
    print(" QUESTION: which COLOUR will you measure now?")
    print(" (First set the lamp to show only this colour.)")
    print()
    for k in sorted(COLOR_PRESETS):
        name, wl = COLOR_PRESETS[k]
        print(f"     {k} = {name:6s} ({wl} nm)")
    print("     4 = Other  (you type the name and wavelength)")
    print("     q = finish and exit the program")
    print()
    print(" Why: each colour is analysed separately. For 1-3 the wavelength")
    print(" is filled in for you, so you cannot mistype it.")
    while True:
        choice = input(" Type 1, 2, 3, 4, or q, then press ENTER: ").strip().lower()
        if choice == "q":
            return None
        if choice in COLOR_PRESETS:
            return COLOR_PRESETS[choice]
        if choice == "4":
            name = ask("   Name of this colour (e.g. 'Amber'): ", "Other")
            wl = ask("   Its nominal peak wavelength in nm (e.g. 590): ", "")
            return (name, wl)
        print("   That key is not on the menu. Please type 1, 2, 3, 4, or q.")


def choose_intensity():
    """Return 'Minimum' or 'Maximum' (or None to go back/finish)."""
    print("\n " + "-" * 60)
    print(" QUESTION: which lamp INTENSITY are you using right now?")
    print(" First set the lamp's brightness with its remote/knob, then choose:")
    print()
    print("     1 = Minimum  (the lamp's LOWEST brightness)")
    print("     2 = Maximum  (the lamp's FULL brightness)")
    print("     b = go back to the colour menu")
    print()
    print(" Why: you will measure each colour at BOTH brightnesses, giving two")
    print(" curves per colour. The program records which one you are doing now.")
    while True:
        c = input(" Type 1, 2, or b, then press ENTER: ").strip().lower()
        if c == "1":
            return "Minimum"
        if c == "2":
            return "Maximum"
        if c == "b":
            return None
        print("   Please type 1, 2, or b.")


def main():
    ser = connect()
    outfile = make_outfile()
    sid = 1

    print("=" * 64)
    print(" AS7341 CALIBRATION + DISTANCE CAPTURE")
    print("=" * 64)
    print()
    print(" WELCOME. This program records light measurements with your sensor.")
    print(" It will tell you exactly what to do at every step. Take your time.")
    print()
    print(" You will NOT write any code. You only need to:")
    print("   - type a number or a word when asked, then press ENTER, and")
    print("   - move the sensor to a distance with your ruler when it asks.")
    print()
    print(" If you get lost: nothing breaks. You can close this window and run")
    print(" the program again - data you already saved is kept.")
    print()
    print(" " + "-" * 60)
    print(" BEFORE YOU START, please make sure:")
    print("   [ ] The reference meter's sensor head is right next to the AS7341.")
    print("   [ ] The lamp has been ON for 10-15 minutes (it needs to warm up).")
    print("   [ ] You took a test reading and nothing was saturating")
    print("       (this program also rejects saturated readings).")
    print(" " + "-" * 60)
    print(f"\n Your measurements will be saved automatically here:")
    print(f"   {outfile}")
    input("\n Press ENTER when you are ready to begin ...")

    print("\n" + "=" * 64)
    print(" QUESTION: which REFERENCE meter are you using?")
    print(" This is the borrowed meter that shows PPFD (e.g. PHOTOBIO or LI-COR).")
    print(" Why: its name is saved with your data so your paper can cite it.")
    ref_name = ask(" Type its name and press ENTER: ", "PHOTOBIO")

    while True:
        chosen = choose_color()
        if chosen is None:
            break
        color, wl = chosen

        intensity = choose_intensity()
        if intensity is None:
            continue   # back to colour menu
        print(f"\n >>> You are measuring: {color} ({wl} nm) at {intensity} intensity.")
        print(f"     Make sure the lamp is set to {color} only, at {intensity} brightness.")

        print("\n Optional: a short note about the sensor's TEMPERATURE at the close,")
        print(" hot distances (for example 'stable' or '34C'). This does NOT change")
        print(" your results. Just press ENTER to skip it.")
        temp_note = ask(" Type a note, or press ENTER to skip: ", "")

        n_d = len(DISTANCES_CM)
        print("\n " + "=" * 60)
        print(f" Now you will measure {n_d} distances for {color} / {intensity}:")
        print(f"   {', '.join(str(x) for x in DISTANCES_CM)} cm")
        print(" The program will walk you through them ONE AT A TIME.")
        print(" " + "=" * 60)

        for i, d in enumerate(DISTANCES_CM, start=1):
            print("\n " + "-" * 60)
            print(f" >>> DISTANCE {i} of {n_d}  ->  {d} cm")
            print(f"     STEP 1: physically MOVE the sensor so it is {d} cm from the")
            print(f"             lamp face. Measure with your ruler/tape. (The program")
            print(f"             does NOT move anything - you do this by hand.)")
            print(f"     STEP 2: look at the {ref_name} screen and read its PPFD number.")
            print(f"             Type THAT number below. (It is the light reading, NOT")
            print(f"             the distance.) If the meter saturates or shows nothing")
            print(f"             usable, type 'skip' instead.")
            ref = input("             Your reading (a number) or 'skip': ").strip()
            if ref.lower() == "skip":
                print("     -> skipped this distance.")
                continue
            try:
                float(ref)
            except ValueError:
                print("     -> that was not a number; skipping this distance.")
                continue

            print(f"     STEP 3: keep the lamp on {color}, take your hands away from the")
            print(f"             sensor (your shadow changes the reading).")
            input("             When ready, press ENTER and stay still ...")

            parts, status = wait_until_stable(ser)
            if status == "saturated":
                print("     -> NOT saved (too bright). Add a PTFE layer or move farther,")
                print("        then measure this distance again.")
                continue
            if status == "error":
                print("     -> NOT saved (could not read the sensor). Try this distance again.")
                continue
            if status == "timeout":
                ans = ask("     The reading would not settle (lamp flicker or a shaky"
                          " table?).\n     Save it anyway? Type y for yes, ENTER for no: ", "n")
                if ans.lower() != "y":
                    print("     -> not saved.")
                    continue

            sid, n = record_point(ser, outfile, sid, color, wl, intensity,
                                   ref_name, ref, d, temp_note)
            print(f"     OK - saved {n} of {REPLICATES} readings at {d} cm.")

        print(f"\n Finished {color} / {intensity}.")
        print(" Next, you can choose the SAME colour at the other intensity,")
        print(" a NEW colour, or type q to finish.")

    ser.close()
    print("\n" + "=" * 64)
    print(f" ALL DONE. Your data is saved here:")
    print(f"   {outfile}")
    if EXPORT_XLSX:
        export_xlsx(outfile)
    print(" Next step: run ../analysis/analyze_calibration.R on this file")
    print(" to obtain the channel profiles and the PPFD calibration.")
    print("=" * 64)


if __name__ == "__main__":
    main()

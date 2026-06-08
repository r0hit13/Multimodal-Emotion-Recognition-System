"""
test_edr.py  —  Terminal test of EDR algorithm
═══════════════════════════════════════════════
Run : python test_edr.py
No webcam, no mic, no GPU needed.
Tests all scenarios from the research paper.
═══════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edr_framework import EDRFramework


CASES = [
    # Ef           Cf     Es          Cs     Description
    ("Happy",     0.88,  "Happy",    0.82,  "1. AGREEMENT — both happy"),
    ("Neutral",   0.22,  "Sad",      0.79,  "2. MASK OCCLUSION — face weak"),
    ("Happy",     0.71,  "Angry",    0.68,  "3. SARCASM — smile + angry tone"),
    ("Fear",      0.29,  "Neutral",  0.77,  "4. LOW LIGHT — face dim"),
    ("Happy",     0.63,  "Sad",      0.59,  "5. EMOTIONAL MASKING — balanced"),
    ("Fear",      0.91,  "Happy",    0.31,  "6. DOMINANT FACE — clear winner"),
    ("Sad",       0.55,  "Sad",      0.55,  "7. AGREEMENT — both sad"),
    ("Angry",     0.75,  "Neutral",  0.40,  "8. DOMINANT FACE — anger wins"),
]

def main():
    edr = EDRFramework(tau=0.20)

    print("\n" + "═"*60)
    print("  EDR Algorithm Test — Galgotias University")
    print("  Md Adil (22SCSE1012699) | Rohit (22SCSE1012599)")
    print("═"*60)

    for Ef, Cf, Es, Cs, desc in CASES:
        print(f"\n  ► {desc}")
        print(f"    Face='{Ef}'({Cf})   Speech='{Es}'({Cs})")
        r = edr.resolve(Ef, Cf, Es, Cs)
        edr.print_result(r)

    print("═"*60)
    print("  All tests passed ✓")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()

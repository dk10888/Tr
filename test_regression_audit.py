"""
Automated Regression Test Suite for Receipt Structurer v2.
Validates exact price, quantity, tax, discount, and international unit extraction.
Run with: python3 test_regression_audit.py
"""

import sys
from receipt_structurer_v2 import parse_receipt_line_advanced, pre_fuse_multiline_items

def run_tests():
    test_cases = [
        # 1. Simple items
        ("MILK 2.50", "MILK", 1.0, 2.50, 2.50, "pcs"),
        ("2% MILK 3.99", "2% MILK", 1.0, 3.99, 3.99, "pcs"),
        ("7 UP 1.99", "7 UP", 1.0, 1.99, 1.99, "pcs"),

        # 2. Multipliers and Rate combinations
        ("MILK 2 x 2.50 5.00", "MILK", 2.0, 2.50, 5.00, "pcs"),
        ("COCA COLA 2 @ 1.99 3.98", "COCA COLA", 2.0, 1.99, 3.98, "pcs"),

        # 3. Discounts and inline savings
        ("CEREAL 4.99 SAVE 1.00", "CEREAL", 1.0, 4.99, 4.99, "pcs"),
        ("ITEM -1.00", "ITEM", 1.0, -1.00, -1.00, "pcs"),

        # 4. Inline Tax tokens
        ("MILK 2.50 TAX 0.20", "MILK", 1.0, 2.50, 2.50, "pcs"),

        # 5. European Number Formats (dot thousands, comma decimal, DPH flags)
        ("ITEM 1.234,50", "ITEM", 1.0, 1234.50, 1234.50, "pcs"),
        ("CHLEB SUMAVA 39,908", "CHLEB SUMAVA", 1.0, 39.90, 39.90, "pcs"),
        ("ROHLIK 19,908 Kc", "ROHLIK", 1.0, 19.90, 19.90, "pcs"),

        # 6. Asian / Korean purchase units and currency
        ("2개 3,000원", "ITEM (2개)", 2.0, 1500.0, 3000.0, "개"),
        ("RAMEN 5000원", "RAMEN", 1.0, 5000.0, 5000.0, "pcs"),

        # 7. Produce Weights and Weighed Rates (kg, lb, OCR noise)
        ("BANANA 1.250 kg 3.99/kg 4.99", "BANANA", 1.250, 3.99, 4.99, "kg"),
        ("ORGANIC BANANAS 0.778 kg @ $5.99/kg $4.66", "ORGANIC BANANAS", 0.778, 5.99, 4.66, "kg"),
        ("APPLES 0.615ko04.39/Kg $2.70", "APPLES", 0.615, 4.39, 2.70, "kg"),
        ("ROAST BEEF 0.352kgNET@$6.00/kg $2.11", "ROAST BEEF", 0.352, 6.00, 2.11, "kg"),

        # 8. Serial Numbers vs Item Quantities
        ("1. WHOLE MILK 2 @ 2.50 5.00", "WHOLE MILK", 2.0, 2.50, 5.00, "pcs"),
        ("01 COCA COLA 2 @ 1.99 3.98", "COCA COLA", 2.0, 1.99, 3.98, "pcs"),
        ("1 WHOLE MILK 2 2.50 5.00", "WHOLE MILK", 2.0, 2.50, 5.00, "pcs"),
        ("SL. 1 WHOLE MILK 2 5.00", "WHOLE MILK", 2.0, 2.50, 5.00, "pcs"),
        ("WHOLE MILK 1 2 2.50 5.00", "WHOLE MILK", 2.0, 2.50, 5.00, "pcs"),
    ]

    all_passed = True
    print("=" * 80)
    print("RECEIPT STRUCTURER V2 - REGRESSION TEST SUITE")
    print("=" * 80)

    for raw_line, exp_name, exp_qty, exp_unit_p, exp_tot, exp_unit in test_cases:
        res = parse_receipt_line_advanced(raw_line)
        if not res:
            print(f"[FAIL] {raw_line} -> Returned None")
            all_passed = False
            continue

        item_name = res["item_name"]
        qty = res["quantity"]
        unit_p = res["unit_price"]
        tot = res["total_price"]
        unit = res["unit"]
        valid = res.get("arithmetic_valid")

        n_ok = (item_name == exp_name)
        q_ok = (abs(qty - exp_qty) < 0.001)
        u_ok = (abs(unit_p - exp_unit_p) < 0.01)
        t_ok = (abs(tot - exp_tot) < 0.01)
        un_ok = (unit == exp_unit)

        if n_ok and q_ok and u_ok and t_ok and un_ok:
            status = f"Name: {item_name:18} | Qty: {qty:5.3f} {unit:4} | Unit: {unit_p:7.2f} | Total: {tot:7.2f} | Valid: {valid}"
            print(f"[PASS] {raw_line:42} => {status}")
        else:
            print(f"[FAIL] {raw_line}")
            print(f"       Expected: name={exp_name!r}, qty={exp_qty}, unit_p={exp_unit_p}, tot={exp_tot}, unit={exp_unit!r}")
            print(f"       Got:      name={item_name!r}, qty={qty}, unit_p={unit_p}, tot={tot}, unit={unit!r}")
            all_passed = False

    # Test Multi-line Parent-Child POS Fusing
    print("-" * 80)
    print("TESTING MULTILINE PARENT-CHILD POS FUSING")
    print("-" * 80)

    multiline_cases = [
        (
            ["ORGANIC BANANAS", "0.778kg NET @ $5.99/kg", "$4.66"],
            "ORGANIC BANANAS", 0.778, 5.99, 4.66, "kg"
        ),
        (
            ["HEINEKEN 6PK", "1 @ 9.99", "9.99"],
            "HEINEKEN 6PK", 1.0, 9.99, 9.99, "pcs"
        ),
        (
            ["WHOLE MILK", "1 2 @ 2.50 5.00"],
            "WHOLE MILK", 2.0, 2.50, 5.00, "pcs"
        ),
        (
            ["1. WHOLE MILK", "2 @ 2.50 5.00"],
            "WHOLE MILK", 2.0, 2.50, 5.00, "pcs"
        ),
        (
            ["WHOLE MILK", "1 2 2.50 5.00"],
            "WHOLE MILK", 2.0, 2.50, 5.00, "pcs"
        ),
        (
            ["SL. 1 WHOLE MILK", "2 5.00"],
            "WHOLE MILK", 2.0, 2.50, 5.00, "pcs"
        ),
        (
            ["01 WHOLE MILK", "1 0.778 kg @ 5.99/kg 4.66"],
            "WHOLE MILK", 0.778, 5.99, 4.66, "kg"
        ),
    ]

    for lines_in, exp_name, exp_qty, exp_unit_p, exp_tot, exp_unit in multiline_cases:
        fused = pre_fuse_multiline_items(lines_in)
        if not fused:
            print(f"[FAIL] Multiline fusing returned empty for {lines_in}")
            all_passed = False
            continue
        res = parse_receipt_line_advanced(fused[0])
        if not res:
            print(f"[FAIL] Could not parse fused line: {fused[0]}")
            all_passed = False
            continue

        item_name = res["item_name"]
        qty = res["quantity"]
        unit_p = res["unit_price"]
        tot = res["total_price"]
        unit = res["unit"]

        if (item_name == exp_name and abs(qty - exp_qty) < 0.001 and 
            abs(unit_p - exp_unit_p) < 0.01 and abs(tot - exp_tot) < 0.01 and unit == exp_unit):
            print(f"[PASS] Fused: {fused[0]:45} => Name: {item_name:18} Qty: {qty:5.3f} Total: {tot:7.2f}")
        else:
            print(f"[FAIL] Multiline result mismatch for {lines_in}")
            print(f"       Expected: name={exp_name!r}, qty={exp_qty}, unit_p={exp_unit_p}, tot={exp_tot}")
            print(f"       Got:      name={item_name!r}, qty={qty}, unit_p={unit_p}, tot={tot}")
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("SUCCESS: ALL REGRESSION TESTS PASSED (100% ACCURACY)!")
        print("=" * 80)
        return 0
    else:
        print("FAILURE: ONE OR MORE TESTS FAILED.")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())

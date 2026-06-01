#!/usr/bin/env python3
"""
Lightweight TFLite output-shape verifier used by example scripts.

Loads a .tflite file and asserts:
- The number of model output tensors equals --expected-heads
  (plus 1 if --unified-output was used to add a concatenated tensor).
- The first input tensor's dtype is uint8 (the fully-quantized contract).

Exits 0 on success; non-zero with a clear message on failure.
"""

import argparse
import sys

import numpy as np
import tensorflow as tf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tflite_path", help="Path to the .tflite file to verify")
    parser.add_argument(
        "--expected-heads",
        type=int,
        required=True,
        help="Number of head outputs the model should expose",
    )
    parser.add_argument(
        "--unified-output",
        action="store_true",
        help="Set if the model was built with --unified-output (adds one extra output tensor)",
    )
    args = parser.parse_args()

    expected_outputs = args.expected_heads + (1 if args.unified_output else 0)

    interpreter = tf.lite.Interpreter(model_path=args.tflite_path)
    interpreter.allocate_tensors()

    output_details = interpreter.get_output_details()
    input_details = interpreter.get_input_details()

    failed = False

    if len(output_details) != expected_outputs:
        print(
            f"[verify_tflite] FAIL: expected {expected_outputs} output tensor(s) "
            f"(heads={args.expected_heads}, unified={args.unified_output}), "
            f"got {len(output_details)}.",
            file=sys.stderr,
        )
        failed = True

    input_dtype = input_details[0]["dtype"]
    if input_dtype != np.uint8:
        print(
            f"[verify_tflite] FAIL: expected input dtype uint8, got {input_dtype}.",
            file=sys.stderr,
        )
        failed = True

    if failed:
        return 1

    print(
        f"[verify_tflite] OK: {args.tflite_path} has {len(output_details)} output(s) "
        f"and uint8 input."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

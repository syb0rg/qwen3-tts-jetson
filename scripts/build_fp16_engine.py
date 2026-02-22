#!/usr/bin/env python3
"""Build FP16 TensorRT engine for code predictor.

No calibration needed — just ONNX → FP16 TRT engine.

Usage:
    python3 build_fp16_engine.py \
        --onnx models/code_pred_layers.onnx \
        -o models/code_pred_layers_fp16.trt
"""
import argparse
import sys

def build_engine(onnx_path, output_path):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
    config.set_flag(trt.BuilderFlag.FP16)

    print("Building FP16 engine...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: Build failed!")
        return False

    data = bytes(serialized)
    with open(output_path, "wb") as f:
        f.write(data)

    size_mb = len(data) / (1024 * 1024)
    print(f"FP16 engine: {output_path} ({size_mb:.1f} MB)")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("-o", "--output", required=True, help="Output engine path")
    args = parser.parse_args()
    success = build_engine(args.onnx, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

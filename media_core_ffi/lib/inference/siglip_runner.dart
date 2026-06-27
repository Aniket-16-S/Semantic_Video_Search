import 'dart:typed_data';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

class SigLipRunner {
  final OrtSession _session;

  SigLipRunner._(this._session);

  static Future<SigLipRunner> load(Uint8List modelBytes) async {
    final sessionOptions = OrtSessionOptions();
    // Do NOT append hardware providers (DirectML) as they lack support for ConvInteger Int8 ops.
    // Fall back to pure CPU execution provider.
    final session = OrtSession.fromBuffer(modelBytes, sessionOptions);
    return SigLipRunner._(session);
  }

  /// Encodes a single RGB frame (resized to 384x384) into a 1152-dim vector.
  List<double> encodeFrame(Float32List pixelData) {
    // SigLIP SO400M typical input: [1, 3, 384, 384] Float32
    final shape = [1, 3, 384, 384];
    final inputOrt = OrtValueTensor.createTensorWithDataList(pixelData, shape);
    final inputs = {'pixel_values': inputOrt};
    final runOptions = OrtRunOptions();

    final outputs = _session.run(runOptions, inputs);

    inputOrt.release();
    runOptions.release();

    // Extract the vector from output
    final outTensor = outputs[0]!.value as List<dynamic>;
    final vector = (outTensor[0] as List<dynamic>).cast<double>();

    for (var element in outputs) {
      element?.release();
    }

    return vector;
  }

  void dispose() {
    _session.release();
  }
}

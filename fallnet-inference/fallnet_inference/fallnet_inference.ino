/*
 * FallNet — Micro-CNN Fall Detection
 * Hardware: Arduino Nano 33 BLE Sense Rev2 (nRF52840)
 * IMU:      BMI270 (accel + gyro @ 50Hz)
 * Model:    micro_cnn_fold_5_int8.tflite (56 KB)
 * Accuracy: 94.71% | Fall_Init Recall: 98.79%
 *
 * Place fallnet_model.h in the same folder as this sketch.
 */

#include <Arduino_BMI270_BMM150.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "fallnet_model.h"

// ─── Configuration ───────────────────────────────────────────────────────────
#define SAMPLE_RATE_HZ      50
#define WINDOW_SIZE         200       // samples per inference window
#define NUM_CHANNELS        6         // ax, ay, az, gx, gy, gz
#define SAMPLE_INTERVAL_US  (1000000 / SAMPLE_RATE_HZ)   // 20,000 µs

// TFLite tensor arena — adjust if you see allocation errors
#define TENSOR_ARENA_KB     80
#define TENSOR_ARENA_SIZE   (TENSOR_ARENA_KB * 1024)

// ─── Label map (must match training) ─────────────────────────────────────────
const char* CLASS_NAMES[] = {
  "Walking",
  "Jogging",
  "Walking_stairs_updown",
  "Stumble_while_walking",
  "Fall_Initiation",       // ← CRITICAL CLASS
  "Impact_Aftermath"
};
#define FALL_INIT_CLASS 4
#define NUM_CLASSES     6

// ─── Z-score normalization params ────────────────────────────────────────────
// These must match the scaler fitted on your training data.
// Load from your scaler.pkl:
//   import pickle, numpy as np
//   sc = pickle.load(open('scaler.pkl','rb'))
//   print(sc.mean_)   # 6 values
//   print(sc.scale_)  # 6 values
const float SCALER_MEAN[NUM_CHANNELS]  = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
const float SCALER_SCALE[NUM_CHANNELS] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
// ^^^ REPLACE THESE with your actual scaler values before deploying ^^^

// ─── Globals ─────────────────────────────────────────────────────────────────
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static float input_buffer[WINDOW_SIZE][NUM_CHANNELS];
static int   sample_count = 0;

const tflite::Model*         model_ptr   = nullptr;
tflite::MicroInterpreter*    interpreter = nullptr;
TfLiteTensor*                input       = nullptr;
TfLiteTensor*                output      = nullptr;

static unsigned long last_sample_us = 0;

// INT8 quantization params (read from model after init)
static float input_scale;
static int   input_zero_point;
static float output_scale;
static int   output_zero_point;

// ─── Setup ───────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("=== FallNet Inference ===");

  // Init IMU
  if (!IMU.begin()) {
    Serial.println("ERROR: IMU init failed");
    while (1);
  }

  // Configure 50Hz — BMI270 default is 100Hz, we downsample in loop
  Serial.print("Accel sample rate: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyro sample rate:  ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  // Init TFLite
  model_ptr = tflite::GetModel(fallnet_model);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema mismatch");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model_ptr, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed");
    while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  // Read INT8 quantization params
  input_scale      = input->params.scale;
  input_zero_point = input->params.zero_point;
  output_scale     = output->params.scale;
  output_zero_point = output->params.zero_point;

  Serial.print("Input tensor:  ");
  Serial.print(input->dims->data[1]);
  Serial.print(" x ");
  Serial.println(input->dims->data[2]);
  Serial.print("Input dtype:   ");
  Serial.println(input->type == kTfLiteInt8 ? "INT8" : "other");
  Serial.println("Model loaded. Collecting samples...");

  last_sample_us = micros();
}

// ─── Loop ────────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = micros();

  // Enforce 50Hz sampling
  if (now - last_sample_us < SAMPLE_INTERVAL_US) return;
  last_sample_us = now;

  float ax, ay, az, gx, gy, gz;

  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) return;

  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);

  // Store raw sample (z-score normalize before inference)
  input_buffer[sample_count][0] = ax;
  input_buffer[sample_count][1] = ay;
  input_buffer[sample_count][2] = az;
  input_buffer[sample_count][3] = gx;
  input_buffer[sample_count][4] = gy;
  input_buffer[sample_count][5] = gz;
  sample_count++;

  if (sample_count < WINDOW_SIZE) return;

  // ── Run inference ──────────────────────────────────────────────────────────
  sample_count = 0;  // reset for next window

  // Fill input tensor: shape [1, 200, 6]
  // Apply z-score normalization + INT8 quantization
  for (int t = 0; t < WINDOW_SIZE; t++) {
    for (int c = 0; c < NUM_CHANNELS; c++) {
      float normalized = (input_buffer[t][c] - SCALER_MEAN[c]) / SCALER_SCALE[c];
      int8_t quantized = (int8_t)(normalized / input_scale + input_zero_point);
      input->data.int8[t * NUM_CHANNELS + c] = quantized;
    }
  }

  unsigned long t_start = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Invoke failed");
    return;
  }
  unsigned long inference_us = micros() - t_start;

  // Dequantize output and find argmax
  float scores[NUM_CLASSES];
  int   best_class = 0;
  float best_score = -1e9f;

  for (int i = 0; i < NUM_CLASSES; i++) {
    scores[i] = (output->data.int8[i] - output_zero_point) * output_scale;
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_class = i;
    }
  }

  // ── Print result ───────────────────────────────────────────────────────────
  Serial.print("[");
  Serial.print(inference_us / 1000.0f, 1);
  Serial.print(" ms] ");
  Serial.print(CLASS_NAMES[best_class]);

  if (best_class == FALL_INIT_CLASS) {
    Serial.print("  *** FALL DETECTED ***");
  }

  Serial.print("  (");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(CLASS_NAMES[i]);
    Serial.print(": ");
    Serial.print(scores[i], 3);
    if (i < NUM_CLASSES - 1) Serial.print(", ");
  }
  Serial.println(")");
}

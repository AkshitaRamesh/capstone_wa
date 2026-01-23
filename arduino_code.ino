// pins
const int emgPin = A0;   // MyoWare analog output -> A0
const int ledPin = 12;   // LED mirrors deadman

// output 
// axes[0]=x, axes[1]=y, axes[2]=z, axes[3]=yaw
int axes[4]    = {0, 0, 0, 0};

// buttons[0..5], where buttons[5] is deadman by default
int buttons[6] = {0, 0, 0, 0, 0, 0};

// ---------- EMG thresholding ----------
const int onThreshold  = 500;  // activate when EMG rises above this
const int offThreshold = 450;  // deactivate when EMG falls below this (hysteresis)

// Simple low-pass smoothing (0..1023)
// higher alpha = smoother but more lag
float emgFiltered = 0.0f;
const float alpha = 0.6f;      // 0.1-0.3 is a common starting range

bool deadmanActive = false;

unsigned long lastPrintMs = 0;
const unsigned long printPeriodMs = 50; // 20 Hz

static inline void computeOutputs(bool active) {
  // Deadman button
  buttons[5] = active ? 1 : 0;

  // LED mirrors deadman
  digitalWrite(ledPin, active ? HIGH : LOW);

  // Always zero other axes
  axes[0] = 0;  // x
  axes[1] = 0;  // y
  axes[2] = 0;  // z

  // Yaw ONLY while deadman active
  axes[3] = active ? 1 : 0;  // yaw
}

static inline void printPacket() {
  // Print EXACTLY 10 values: 4 axes + 6 buttons
  Serial.print(axes[0]);
  for (int i = 1; i < 4; i++) {
    Serial.print(",");
    Serial.print(axes[i]);
  }
  for (int i = 0; i < 6; i++) {
    Serial.print(",");
    Serial.print(buttons[i]);
  }
  Serial.println();
}

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  unsigned long now = millis();

  // Read EMG and filter
  int emgRaw = analogRead(emgPin);
  emgFiltered = alpha * emgRaw + (1.0f - alpha) * emgFiltered;

  // Threshold with hysteresis to avoid chatter
  if (!deadmanActive && emgFiltered >= onThreshold) {
    deadmanActive = true;
  } else if (deadmanActive && emgFiltered <= offThreshold) {
    deadmanActive = false;
  }

  // Print at fixed rate (20 Hz)
  if ((now - lastPrintMs) >= printPeriodMs) {
    lastPrintMs = now;

    computeOutputs(deadmanActive);
    printPacket();

    // Optional debug (comment out if your ROS parser expects exactly 10 values)
    // Serial.print(" EMGraw="); Serial.print(emgRaw);
    // Serial.print(" EMGf="); Serial.println(emgFiltered);
  }
}

use nalgebra::Vector3;

pub struct PIDController {
    kp: f32,
    ki: f32,
    kd: f32,
    integral: Vector3<f32>,
    prev_error: Vector3<f32>,
}

impl PIDController {
    pub fn new(kp: f32, ki: f32, kd: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: Vector3::zeros(),
            prev_error: Vector3::zeros(),
        }
    }

    pub fn step(&mut self, error: Vector3<f32>) -> Vector3<f32> {
        self.integral += error * 0.01;
        let derivative = (error - self.prev_error) / 0.01;
        self.prev_error = error;

        self.kp * error + self.ki * self.integral + self.kd * derivative
    }
}

pub fn send_pwm(output: Vector3<f32>) {
    // Write to PWM registers
}

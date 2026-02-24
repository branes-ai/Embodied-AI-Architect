use nalgebra::Vector3;

mod sensor;
mod control;

fn main() {
    let mut controller = control::PIDController::new(1.0, 0.1, 0.05);

    loop {
        let accel = sensor::read_accelerometer();
        let gyro = sensor::read_gyroscope();

        let fused = sensor::fuse_imu(accel, gyro);
        let output = controller.step(fused);

        control::send_pwm(output);
    }
}

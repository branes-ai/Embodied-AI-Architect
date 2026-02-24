use nalgebra::Vector3;

pub fn read_accelerometer() -> Vector3<f32> {
    // Read from I2C accelerometer
    Vector3::new(0.0, 0.0, 9.81)
}

pub fn read_gyroscope() -> Vector3<f32> {
    // Read from I2C gyroscope
    Vector3::new(0.0, 0.0, 0.0)
}

pub fn fuse_imu(accel: Vector3<f32>, gyro: Vector3<f32>) -> Vector3<f32> {
    // Complementary filter for sensor fusion
    let alpha = 0.98;
    accel * (1.0 - alpha) + gyro * alpha
}

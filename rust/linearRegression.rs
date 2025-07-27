struct LinearRegression {
    slope: f64,
    intercept: f64,
}

impl LinearRegression {
    /// Create a new LinearRegression model
    fn new() -> Self {
        LinearRegression {
            slope: 0.0,
            intercept: 0.0,
        }
    }

    /// Fit the model to the training data using least squares
    fn fit(&mut self, x: &Vec<f64>, y: &Vec<f64>) {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;

        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();

        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..x.len() {
            numerator += (x[i] - mean_x) * (y[i] - mean_y);
            denominator += (x[i] - mean_x).powi(2);
        }

        self.slope = numerator / denominator;
        self.intercept = mean_y - self.slope * mean_x;
    }

    /// Predict a value given x using the trained model
    fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    /// Predict multiple values at once
    fn predict_batch(&self, x_vals: &Vec<f64>) -> Vec<f64> {
        x_vals.iter().map(|&x| self.predict(x)).collect()
    }
}


fn main() {
    let x_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_vals = vec![3.0, 5.0, 7.0, 9.0, 11.0];

    let mut model = LinearRegression::new();
    model.fit(&x_vals, &y_vals);

    println!("Slope: {:.4}", model.slope);
    println!("Intercept: {:.4}", model.intercept);

    let x_new = 6.0;
    let prediction = model.predict(x_new);
    println!("Prediction for x = {}: y = {:.4}", x_new, prediction);

    // Batch predictions
    let batch_preds = model.predict_batch(&vec![6.0, 7.0, 8.0]);
    println!("Batch predictions: {:?}", batch_preds);
}

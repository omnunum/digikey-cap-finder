import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class CapacitorFeatures:
    """Physical characteristics of a capacitor for ESR prediction."""
    capacitance: float  # in μF
    voltage: float      # in V
    dissipation_factor: float  # unitless
    diameter: float     # mm
    length: float       # mm
    volume: float       # mm³
    esr_100khz: Optional[float] = None  # Ω
    # These are only for display purposes, not used in prediction
    series: Optional[str] = None
    manufacturer: Optional[str] = None


class CapacitorESRModel:
    """Model for predicting ESR values at 100kHz based on capacitor physical characteristics."""
    
    def __init__(self):
        self.model = None
        self.feature_columns: List[str] = [
            'Capacitance', 'Voltage', 'Dissipation Factor', 
            'Volume', 'Diameter', 'Length'
        ]
        self.target_column: str = 'ESR_100kHz'
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess capacitor data from CSV file."""
        df = pd.read_csv(csv_path)
        
        # Rename columns for consistent access
        df = df.rename(columns={
            'Case Size Diameter': 'Diameter',
            'Case Size Length': 'Length',
            'ESR/Z 20°C@100kHz': 'ESR_100kHz'
        })
        
        # Calculate volume based on cylindrical dimensions
        df['Volume'] = np.pi * (df['Diameter']/2)**2 * df['Length']
        
        # Convert all numeric columns to float
        numeric_columns = ['Capacitance', 'Voltage', 'Dissipation Factor', 
                          'Diameter', 'Length', 'Volume', 'ESR_100kHz']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and target from dataframe, removing rows with missing target values.
        IMPORTANT: Rows without 100kHz ESR ratings are completely removed from training.
        """
        # Count before filtering
        initial_count = len(df)
        
        # Explicitly remove rows with missing 100kHz ESR values
        valid_df = df.dropna(subset=[self.target_column])
        esr_dropped_count = initial_count - len(valid_df)
        
        print(f"Total rows: {initial_count}")
        print(f"Rows with missing 100kHz ESR (dropped): {esr_dropped_count} ({esr_dropped_count/initial_count*100:.1f}%)")
        print(f"Rows with valid 100kHz ESR: {len(valid_df)} ({len(valid_df)/initial_count*100:.1f}%)")
        
        # Now check for missing feature values
        feature_valid_df = valid_df.dropna(subset=self.feature_columns)
        feature_dropped_count = len(valid_df) - len(feature_valid_df)
        
        if feature_dropped_count > 0:
            print(f"Additionally dropped {feature_dropped_count} rows with missing feature values")
            
        # Final valid dataset for training
        X = feature_valid_df[self.feature_columns].values
        y = feature_valid_df[self.target_column].values.astype(np.float64)
        
        print(f"Final training dataset: {len(X)} rows")
        
        return X, y
    
    def train(self, csv_path: str) -> Dict[str, Any]:
        """Train the ESR prediction model using data points with valid 100kHz measurements."""
        df = self.load_data(csv_path)
        X, y = self.prepare_features(df)
        
        if len(X) < 10:
            print("Warning: Very few valid training samples available. Model may not be reliable.")
            if len(X) < 2:
                raise ValueError("Insufficient data points with 100kHz ESR values for training.")
        
        # Count unique series in training data to verify generalization
        if 'Series' in df.columns:
            valid_df = df.dropna(subset=[self.target_column])
            unique_series = valid_df['Series'].nunique()
            print(f"Training on data from {unique_series} different capacitor series")
            
            # Print series distribution to help understand data composition
            series_counts = valid_df['Series'].value_counts()
            print("Series distribution in training data:")
            for series, count in series_counts.head(5).items():
                print(f"  {series}: {count} samples")
            if len(series_counts) > 5:
                print(f"  ... and {len(series_counts) - 5} more series")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0)
        }
        
        best_model = None
        best_score = -np.inf
        best_metrics = {}
        
        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'model_name': name,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
            
            print(f"Model: {name}, RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = pipeline
                best_metrics = metrics
        
        self.model = best_model
        
        # Feature importance analysis for RandomForest model
        if best_model is not None and 'RandomForest' in best_metrics['model_name']:
            rf_model = best_model.named_steps['model']
            feature_importances = rf_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            for _, row in importance_df.iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")
            
            best_metrics['feature_importance'] = importance_df.to_dict('records')
        
        return best_metrics
    
    def predict_esr_100khz(self, capacitor: CapacitorFeatures) -> float:
        """Predict ESR at 100kHz using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
            
        features = np.array([[
            capacitor.capacitance,
            capacitor.voltage,
            capacitor.dissipation_factor,
            capacitor.volume,
            capacitor.diameter,
            capacitor.length
        ]])
        
        prediction = float(self.model.predict(features)[0])
        
        # Ensure the prediction is positive
        return max(0.001, prediction)


def extract_capacitor_features(row: pd.Series) -> CapacitorFeatures:
    """Extract capacitor features from a dataframe row."""
    volume = np.pi * (float(row['Diameter'])/2)**2 * float(row['Length'])
    
    # Handle potentially missing ESR values
    esr_100khz = None
    if 'ESR_100kHz' in row and pd.notnull(row['ESR_100kHz']):
        esr_100khz = float(row['ESR_100kHz'])
    
    return CapacitorFeatures(
        capacitance=float(row['Capacitance']),
        voltage=float(row['Voltage']),
        dissipation_factor=float(row['Dissipation Factor']),
        diameter=float(row['Diameter']),
        length=float(row['Length']),
        volume=volume,
        esr_100khz=esr_100khz,
        # These are kept only for display purposes, not for prediction
        series=str(row['Series']) if 'Series' in row else None,
        manufacturer=str(row['Manufacturer']) if 'Manufacturer' in row else None
    )


def generate_comparison_chart(predictions: List[float], actuals: List[float], series_names: List[str]) -> None:
    """Generate a chart comparing predicted vs actual ESR values at 100kHz."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(actuals, predictions, s=50, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    margin = (max_val - min_val) * 0.1
    plt.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'r--', label='Perfect prediction')
    
    # Add labels and title
    plt.xlabel('Actual ESR (Ω)')
    plt.ylabel('Predicted ESR (Ω)')
    plt.title('Predicted vs Actual 100kHz ESR Values')
    plt.grid(True)
    
    # Only add annotations if there are a reasonable number of points
    if len(actuals) <= 20:
        for actual, predicted, series in zip(actuals, predictions, series_names):
            plt.annotate(
                series, 
                (actual, predicted),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
    
    # Calculate statistics
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    avg_error = sum(errors) / len(errors)
    error_percentages = [100 * err / act if act > 0 else float('nan') for err, act in zip(errors, actuals)]
    avg_pct_error = np.nanmean(error_percentages)
    
    # Add stats annotation
    plt.figtext(0.02, 0.02, 
               f"Number of samples: {len(actuals)}\n"
               f"Average absolute error: {avg_error:.3f}Ω\n"
               f"Average percentage error: {avg_pct_error:.1f}%",
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('esr_prediction_comparison.png')
    print("Comparison chart saved as 'esr_prediction_comparison.png'")


def main():
    """Train model and demonstrate ESR prediction."""
    csv_path = 'all_series_priority_data.csv'
    
    model = CapacitorESRModel()
    metrics = model.train(csv_path)
    
    print(f"\nBest model: {metrics['model_name']}")
    print(f"R² score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    # Load data again for demonstration
    df = model.load_data(csv_path)
    valid_samples = df.dropna(subset=['ESR_100kHz'])
    
    if not valid_samples.empty:
        # For table display, limit to a few samples
        display_count = min(5, len(valid_samples))
        display_samples = valid_samples.sample(display_count) if len(valid_samples) > display_count else valid_samples
        
        # For visualization, use all valid samples
        all_predictions = []
        all_actuals = []
        all_series = []
        
        # First generate all predictions (for both table and visualization)
        for _, row in valid_samples.iterrows():
            capacitor = extract_capacitor_features(row)
            predicted_esr = model.predict_esr_100khz(capacitor)
            
            all_predictions.append(predicted_esr)
            all_actuals.append(capacitor.esr_100khz)
            all_series.append(str(capacitor.series))
        
        # Print table with a limited number of samples
        print("\n100kHz ESR Predictions (Sample):")
        print("-" * 50)
        print(f"{'Series':<10} {'Cap (μF)':<10} {'Voltage (V)':<12} {'Dims (mm)':<15} {'Predicted (Ω)':<15} {'Actual (Ω)':<15} {'Error (%)':<10}")
        print("-" * 50)
        
        for i, (_, row) in enumerate(display_samples.iterrows()):
            capacitor = extract_capacitor_features(row)
            predicted_esr = model.predict_esr_100khz(capacitor)
            
            dims = f"{capacitor.diameter}×{capacitor.length}"
            error_pct = abs(predicted_esr - capacitor.esr_100khz) / capacitor.esr_100khz * 100 if capacitor.esr_100khz else float('nan')
            
            print(f"{str(capacitor.series):<10} {capacitor.capacitance:<10.1f} {capacitor.voltage:<12.1f} {dims:<15} {predicted_esr:<15.3f} {capacitor.esr_100khz:<15.3f} {error_pct:<10.1f}")
        
        # Print summary statistics
        errors = [abs(p - a) for p, a in zip(all_predictions, all_actuals)]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        
        error_percentages = [100 * err / act if act > 0 else float('nan') for err, act in zip(errors, all_actuals)]
        avg_pct_error = np.nanmean(error_percentages)
        max_pct_error = np.nanmax(error_percentages)
        
        print("\nOverall Prediction Statistics:")
        print(f"Total samples evaluated: {len(all_predictions)}")
        print(f"Average absolute error: {avg_error:.3f}Ω")
        print(f"Maximum absolute error: {max_error:.3f}Ω")
        print(f"Average percentage error: {avg_pct_error:.1f}%")
        print(f"Maximum percentage error: {max_pct_error:.1f}%")
        
        # Generate comparison visualization with all samples
        try:
            generate_comparison_chart(all_predictions, all_actuals, all_series)
        except ImportError:
            print("\nMatplotlib not available - skipping chart generation")


if __name__ == "__main__":
    main() 
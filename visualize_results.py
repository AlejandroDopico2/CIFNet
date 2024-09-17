import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_dataframe(results):
    data = []
    
    for exp in results:
        data.append(exp)
    
    return pd.DataFrame(data)

def analyze_best_hyperparameters(df, output_dir):
    best_params = []
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # Mejor hiperparámetro basado en Test Accuracy
        best_test_accuracy_row = dataset_df.loc[dataset_df['test_accuracy'].idxmax()]
        
        # Mejor hiperparámetro basado en el ratio Train/Test Accuracy
        dataset_df['train_test_ratio'] = dataset_df['train_accuracy'] / dataset_df['test_accuracy']
        best_train_test_ratio_row = dataset_df.loc[dataset_df['train_test_ratio'].idxmin()]
        
        best_params.append({
            'dataset': dataset,
            'best_test_accuracy': best_test_accuracy_row.to_dict(),
            'best_train_test_ratio': best_train_test_ratio_row.to_dict()
        })

    print(f"Best hyperparameters saved to: {os.path.join(output_dir, 'best_hyperparameters.json')}")

    
    return best_params

def save_best_hyperparameters(best_params, output_dir):
    best_params_file = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Best hyperparameters saved to: {best_params_file}")

def plot_accuracy_comparison(df, output_dir):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='dataset', y='test_accuracy', data=df)
    plt.title('Test Accuracy Comparison Across Datasets')
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()

def plot_parameter_influence(df, output_dir):
    params = ['backbone', 'batch_size', 'rolann_lambda', 'learning_rate', 'dropout']
    
    for param in params:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=param, y='test_accuracy', data=df)
        plt.title(f'Influence of {param} on Test Accuracy')
        plt.savefig(os.path.join(output_dir, f'{param}_influence.png'))
        plt.close()

def plot_learning_rate_vs_accuracy(df, output_dir):
    plt.figure(figsize=(12, 6))
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        plt.scatter(dataset_df['learning_rate'], dataset_df['test_accuracy'], label=dataset)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Accuracy')
    plt.title('Learning Rate vs Test Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'learning_rate_vs_accuracy.png'))
    plt.close()

def plot_heatmap(df, output_dir):
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        pivot = dataset_df.pivot_table(values='test_accuracy', 
                                       index='learning_rate', 
                                       columns='rolann_lambda', 
                                       aggfunc='mean')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu')
        plt.title(f'{dataset}: Test Accuracy Heatmap (Learning Rate vs ROLANN Lambda)')
        plt.savefig(os.path.join(output_dir, f'{dataset}_heatmap.png'))
        plt.close()

def main():
    results_file = 'experiment_results.json'  # Update this path
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_file)
    df = create_dataframe(results)

    plot_accuracy_comparison(df, output_dir)
    plot_parameter_influence(df, output_dir)
    plot_learning_rate_vs_accuracy(df, output_dir)
    plot_heatmap(df, output_dir)

    # Obtener y guardar los mejores hiperparámetros
    best_params = analyze_best_hyperparameters(df, output_dir)
    save_best_hyperparameters(best_params, output_dir)

    print(f"Visualizations have been saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
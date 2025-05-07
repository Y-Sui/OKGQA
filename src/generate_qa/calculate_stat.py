import tiktoken
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ..config.generate_qa_config import PLOTS_DIR, PATHS


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def calculate_stats(df):
    """
    Calculate the token length of the questions, the number of questions, the number of unique dbpedia entities, 
    the number of questions for each type, the number of questions for each naturalness, the number of questions for each difficulty
    """
    number_of_questions = len(df)

    # Count the number of unique DBpedia entities
    unique_dbpedia_entities = set()
    for placeholder in df["placeholders"]:
        unique_dbpedia_entities.update(placeholder.values())
        
    print(unique_dbpedia_entities)

    # Display results
    print(f"Number of questions: {number_of_questions}")
    print(f"Number of unique DBpedia entities: {len(unique_dbpedia_entities)}")

    avg_token_question = 0
    for question in df["question"]:
        num_token = num_tokens_from_messages([{"content": question}])
        avg_token_question += num_token

    avg_token_question = avg_token_question / number_of_questions
    print(f"Average token length of questions: {avg_token_question}")


    # Count the number of questions for each type
    type_counts = df["type"].value_counts()
    # Count the number of questions for each naturalness
    type_naturalness_counts = df.groupby(['type', 'naturalness']).size().unstack()
    # Count the number of questions for each difficulty
    type_difficulty_counts = df.groupby(['type', 'difficulty']).size().unstack()
    
    print(type_counts)
    print(type_naturalness_counts)
    print(type_difficulty_counts)
    
    return type_counts, type_naturalness_counts, type_difficulty_counts


def plot_stats(type_counts, type_naturalness_counts, type_difficulty_counts):
    """
    Plot the statistics of the generated dataset.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # set the style of the plots
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 10})

    plt.figure(figsize=(5, 3))
    sns.barplot(x=type_counts.index, y=type_counts.values, hue=type_counts.index, palette='viridis', legend=False)
    plt.title('Count of Each Type')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'type_counts.png'), dpi=300)
    plt.show()

    # plot the distribution of naturalness by type
    plt.figure(figsize=(5, 3))
    ax1 = type_naturalness_counts.plot(
        kind='bar', stacked=True, colormap='viridis', edgecolor='w'
    )
    plt.title('Naturalness Distribution by Type')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(
        title='Naturalness', loc='upper right', bbox_to_anchor=(1, 1.05), framealpha=0.4
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'naturalness_distribution.png'), dpi=300)
    plt.show()

    # plot the distribution of difficulty by type
    plt.figure(figsize=(5, 3))
    ax2 = type_difficulty_counts.plot(
        kind='bar', stacked=True, colormap='plasma', edgecolor='w'
    )
    plt.title('Difficulty Distribution by Type')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(
        title='Difficulty', loc='upper right', bbox_to_anchor=(1, 1.05), framealpha=0.4
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'difficulty_distribution.png'), dpi=300)
    plt.show()
    
    
def plot_pan_stats(type_counts):
    """
    Plot the PAN distribution of the generated dataset.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # Data for the chart
    categories = [
        'Cause Explanation', 
        'Outcome Prediction', 
        'Contrast Analysis', 
        'Relationship Explanation', 
        'Evaluation and Reflection', 
        'Application and Practice', 
        'Character Description', 
        'Trend Prediction',
        'Historical Comparison', 
        'Event Description'
    ]
    colors = [
        '#ffcc99', 
        '#ff9966', 
        '#ff6666',
        '#ff99cc', 
        '#66b3ff',
        '#33cccc', 
        '#99cc33', 
        '#ccff66', 
        '#ffcc33', 
        '#cc3300'
    ]
    
    # Filter out categories with zero values
    non_zero_data = []
    non_zero_categories = []
    non_zero_colors = []
    
    for i, category in enumerate(categories):
        value = type_counts.get(category.lower(), 0)
        if value > 0:
            non_zero_data.append(value)
            non_zero_categories.append(category)
            non_zero_colors.append(colors[i])
    
    if not non_zero_data:
        print("Warning: No data to plot in pie chart")
        return
    
    plt.figure(figsize=(7, 7))
    plt.pie(
        non_zero_data, 
        labels=non_zero_categories, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=non_zero_colors, 
        wedgeprops={'width': 0.3, 'edgecolor': 'w'},
        explode=[0.003]*len(non_zero_data)
    )
    plt.savefig(os.path.join(PLOTS_DIR, 'pan_distribution.png'), dpi=300)
    print(f"Pan distribution saved to {os.path.join(PLOTS_DIR, 'pan_distribution.png')}")
    plt.show()
    
    
def main():
    dataset_name = os.path.join(PATHS["queries_dir"], "questions_20250507_100_post_processed.csv")
    df = pd.read_csv(dataset_name)    
    type_counts, type_naturalness_counts, type_difficulty_counts = calculate_stats(df)
    plot_stats(type_counts, type_naturalness_counts, type_difficulty_counts)
    plot_pan_stats(type_counts)

if __name__ == "__main__":
    main()
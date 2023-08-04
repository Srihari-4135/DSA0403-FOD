import string

def process_text(text):
    # Remove punctuation and convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()

    # Split the text into individual words
    words = text.split()

    return words

def calculate_frequency(words):
    # Create a dictionary to store word frequency
    word_freq = {}

    # Iterate through each word and update the word frequency in the dictionary
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

    return word_freq

def display_frequency_distribution(word_freq):
    # Display the word frequency distribution
    print("Word Frequency Distribution:")
    for word, freq in word_freq.items():
        print(f"{word}: {freq}")

if __name__ == "__main__":
    # Replace the file path with the provided URL to the text file
    file_path = "https://raw.githubusercontent.com/Muttamatam-Sreeharsha-0471/Data-Science-programs/main/sample_text.txt"

    # Read the text from the specified URL
    import urllib.request
    with urllib.request.urlopen(file_path) as url:
        text = url.read().decode()

    # Process the text
    words = process_text(text)

    # Calculate word frequency
    word_freq = calculate_frequency(words)

    # Display the word frequency distribution
    display_frequency_distribution(word_freq)

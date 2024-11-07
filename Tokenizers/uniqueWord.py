def count_unique_words(file_path):
    # Open the text file in read mode
    with open(file_path, 'r') as file:
        # Read the content of the file
        text = file.read()
        
        # Convert the text to lowercase to ensure case insensitivity
        text = text.lower()
        
        # Remove punctuation using list comprehension and join method
        text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
        
        # Split the text into words
        words = text.split()
        
        # Use a set to find unique words
        unique_words = set(words)
        
        # Return the number of unique words
        return len(unique_words)

# Example usage
file_path = 'sentences.txt'  # Replace with your file path
unique_word_count = count_unique_words(file_path)
print(f'Number of unique words: {unique_word_count}')

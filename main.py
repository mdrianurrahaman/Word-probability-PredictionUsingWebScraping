import requests
from bs4 import BeautifulSoup
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize

######################### get all text and split it into sentence ############################################## 


def scrape_text_from_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text_elements = soup.find_all(text=True)
            extracted_text = ' '.join(text_elements)
            return extracted_text
        else:
            print(f"Failed to retrieve the content. Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def split_into_sentences(text):
    # Define sentence-ending punctuation marks
    sentence_endings = ['.', '!', '?']
    sentences = []
    current_sentence = ""

    for char in text:
        current_sentence += char
        if char in sentence_endings:
            sentences.append(current_sentence.strip())
            current_sentence = ""

    # Add the last sentence if there's any remaining text
    if current_sentence:
        sentences.append(current_sentence.strip())

    return sentences

#website_url = 'https://www.google.com/search?q=business+india+today&oq=&gs_lcrp=EgZjaHJvbWUqCQgDECMYJxjqAjIJCAAQIxgnGOoCMgkIARAjGCcY6gIyCQgCECMYJxjqAjIJCAMQIxgnGOoCMgkIBBAjGCcY6gIyCQgFECMYJxjqAjIJCAYQIxgnGOoCMgkIBxAjGCcY6gLSAQkyMjI1ajBqMTWoAgiwAgE&sourceid=chrome&ie=UTF-8'
website_url='https://www.wakefit.co/'
extracted_text = scrape_text_from_website(website_url)

if extracted_text:
    sentences = split_into_sentences(extracted_text)
    for idx, sentence in enumerate(sentences, start=1):
        print(f"Sentence {idx}: {sentence}")


######################## probab ############################ 


def calculate_bigram_probabilities(text):
    #words = extracted_text.lowercase().split()
    words=text.lower().split()
    print(words)

    count = defaultdict(lambda: defaultdict(int))
    word_count = defaultdict(int)
    
    
    for i in range(len(words) - 1):
        prev_word = words[i]
        next_word = words[i + 1]
        count[prev_word][next_word] += 1
        word_count[prev_word] += 1
    
    count_probabilities = defaultdict(dict)
    for prev_word, next_words in count.items():
        total_next_words = sum(next_words.values())
        for next_word, count in next_words.items():
            probability = count / total_next_words
            count_probabilities[prev_word][next_word] = probability
    
    return count_probabilities, word_count

# Example text
#extracted_text = "Rian is a good boy . He plays football . He chooses cse stream."
extracted_text = scrape_text_from_website(website_url)
# Calculate bigram probabilities and word counts
count_probabilities, word_count = calculate_bigram_probabilities(extracted_text)

# Get the list of unique words
unique_words = list(word_count.keys())


# Initialize the matrix with zeros
matrix = np.zeros((len(unique_words), len(unique_words)))

# Fill in the matrix with bigram probabilities
for i, prev_word in enumerate(unique_words):
    for j, next_word in enumerate(unique_words):
        if next_word in count_probabilities[prev_word]:
            matrix[i, j] = count_probabilities[prev_word][next_word]

# Normalize the matrix row-wise to ensure the probabilities sum up to 1
normalized_matrix = normalize(matrix, norm='l1', axis=1)

# Print the normalized matrix
print("Conditional Probability Matrix:")
print(normalized_matrix) 

np.save('normalized_matrix.npy',normalized_matrix)
np.save('unique_words.npy',np.array(unique_words))
loaded_normalized_matrix = np.load('normalized_matrix.npy')
loaded_unique_words = np.load('unique_words.npy')

input_word = input("Enter a word: ").lower()

if input_word in loaded_unique_words:
    word_index = loaded_unique_words.tolist().index(input_word)
    probabilities = loaded_normalized_matrix[word_index]
    print(f"Word: {input_word}")
    print("Probabilities of the word following each unique word:")
    for i, word in enumerate(loaded_unique_words):
        print(f"{word}: {probabilities[i]:.4f}")
else:
    print(f"The word '{input_word}' is not in the vocabulary.")


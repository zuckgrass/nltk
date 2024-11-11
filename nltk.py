import nltk
from nltk.tokenize import word_tokenize, sent_tokenize  # Ensure word_tokenize is imported
from nltk.corpus import stopwords
from nltk import pos_tag

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


#Task 1

print("Task 1")
# Sample text variable with 5 sentences
text = "This is the first sentence. Here is the second one. This is the third sentence. The fourth sentence is here. Finally, this is the fifth."

# Ensure required resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# a. Count the number of words in the text
words = word_tokenize(text)
word_count = len(words)
print(f'Number of words: {word_count}')

# b. Words starting with a vowel and their count
vowel_words = [word for word in words if word.lower()[0] in 'aeiou']
print(f'Words starting with a vowel: {vowel_words}, Count: {len(vowel_words)}')

# c. Words starting with a consonant
consonant_words = [word for word in words if word.isalpha() and word.lower()[0] not in 'aeiou']
print(f'Words starting with a consonant: {consonant_words}')

# d. Choose three words and find their positions
sample_words = ['first', 'second', 'fourth']
positions = {word: [i for i, w in enumerate(words) if w.lower() == word] for word in sample_words}
print(f'Positions of selected words: {positions}')

# e. Replace a word with your surname
modified_text = text.replace('first', 'Pisotska')
print(f'Modified text: {modified_text}')

#Task 2

print("Task 2")
# Given text with 6 words
text_2 = "Programming is fun because I know"

# Create a list of characters without spaces
char_list = list(text_2.replace(" ", ""))
print(f'List of characters: {char_list}')

# Get the first two letters of each word
first_two_chars = [word[:2] for word in text_2.split()]
print(f'First two letters of each word: {first_two_chars}')

# Create a new list excluding 'a' and 'b'
filtered_chars = [char for char in char_list if char not in ['a', 'b']]
print(f'New list without "a" and "b": {filtered_chars}')

#Task 3

print("Task 3")
text_3 = "Today I learned Python, Java, C++, JavaScript, and Ruby."

# Tokenize the text and filter programming languages
programming_languages = ['Python', 'Java', 'C++', 'JavaScript', 'Ruby']
found_languages = [lang for lang in programming_languages if lang in text_3]
print(f'Programming languages found: {found_languages}')

#Task 4

print("Task 4")
email_text = "ivan@example.com, Maria@gmail.com, Petro@yahoo.com"

# Split email addresses and extract domains
domains = [email.split('@')[1] for email in email_text.split(', ')]
print(f'Domains of email addresses: {domains}')

#Task 5

print("Task 5")
number_text = "333334 333 123 2334 11 222 44 111"
# Tokenize the input text
numbers = word_tokenize(number_text)

# Function to check if a number has repeating digits of length 2 or 3
def has_repeating_digits(num):
    return (len(num) == 2 or len(num) == 3) and all(digit == num[0] for digit in num)

# Extract numbers with repeating digit patterns of length 2 or 3
pattern_numbers = [num for num in numbers if has_repeating_digits(num)]

print(f'Numbers with repeating digits (length 2-3): {pattern_numbers}')

#Task 6
print("Task 6")

datetime_string = "12:34:56 2024-11-04"

# Split time and date
time, date = datetime_string.split()
print(f'Time: {time}, Date: {date}')

# Get hours and year
hour = time.split(':')[0]
year = date.split('-')[0]
print(f'Hour: {hour}, Year: {year}')

#Task 7
print("Task 7")
postal_code = "12345"

# Validate Ukrainian postal code format (5 digits)
is_valid = postal_code.isdigit() and len(postal_code) == 5
print(f'Is the postal code valid? {is_valid}')

#Task 8
print("Task 8")

# Make sure to download the required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Sample text with at least 5 sentences
text = """
Artificial Intelligence (AI) is transforming industries.
It has the potential to improve efficiency and productivity.
However, there are concerns about its impact on jobs.
Many experts believe that AI can enhance human capabilities.
The future of AI is both exciting and uncertain.
"""

# Step 1: Tokenization into sentences
sentences = sent_tokenize(text)
print("Sentences:")
print(sentences)

# Step 2: Tokenization into words
words = word_tokenize(text)
print("\nWords (before removing punctuation and stop words):")
print(words)

# Step 3: Remove punctuation
words_cleaned = [word for word in words if word.isalnum()]
print("\nWords (after removing punctuation):")
print(words_cleaned)

# Step 4: Remove stop words
stop_words = set(stopwords.words('english'))
words_filtered = [word for word in words_cleaned if word.lower() not in stop_words]
print("\nWords (after removing stop words):")
print(words_filtered)

# Step 5: Part of speech tagging
pos_tags = pos_tag(words_filtered)
print("\nPart of Speech Tags:")
print(pos_tags)

# Step 6: Analyzing parts of speech correctness
# Here we will just print the POS tags since further analysis would require specific context or definitions
print("\nAnalysis of Parts of Speech:")
for word, tag in pos_tags:
    if tag.startswith('NN'):
        print(f"{word}: Noun")
    elif tag.startswith('VB'):
        print(f"{word}: Verb")
    elif tag.startswith('JJ'):
        print(f"{word}: Adjective")
    else:
        print(f"{word}: Other (Tag: {tag})")

#Task 9
print("Task 9")

# a. Count the number of words in each sentence
word_count_per_sentence = [len(word_tokenize(sentence)) for sentence in sentences]
print("Number of words in each sentence:")
for i, count in enumerate(word_count_per_sentence, 1):
    print(f"Sentence {i}: {count} words")

# b. Count the number of stop words in the text
stop_word_count = len([word for word in words_cleaned if word.lower() in stop_words])
print(f"\nTotal number of stop words in the text: {stop_word_count}")

# c. Find the longest word
longest_word = max(words_filtered, key=len) if words_filtered else ""
print(f"The longest word in the text: {longest_word}")

# d. Count how many words contain exactly 4 characters
four_char_word_count = len([word for word in words_filtered if len(word) == 4])
print(f"Number of words with exactly 4 characters: {four_char_word_count}")

#Task 10
print("Task 10")
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
# Initialize Stemmer and Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Perform stemming
stemmed_words = [stemmer.stem(word) for word in words_cleaned]
print("Stemmed words:")
print(stemmed_words)

# Perform lemmatization
lemmatized_words = [lemmatizer.lemmatize(word) for word in words_cleaned]
print("\nLemmatized words:")
print(lemmatized_words)

# Compare results
print("\nComparison of Stemming and Lemmatization:")
comparison = list(zip(words_cleaned, stemmed_words, lemmatized_words))
for original, stemmed, lemmatized in comparison:
    print(f"Original: {original}, Stemmed: {stemmed}, Lemmatized: {lemmatized}")

#Task 11
print("Task 11")
import nltk
from nltk.stem import SnowballStemmer
import string
# Read text and stop words from files
with open("text_ukr.txt", "r", encoding="utf-8") as file:
    text = file.read()

with open("stop_words_ukr.txt", "r", encoding="utf-8") as file:
    stop_words_ukr = set(file.read().splitlines())

# Tokenize the text by words
words = word_tokenize(text)

# Remove punctuation
words_cleaned = [word for word in words if word.isalnum()]

# Count the number of punctuation marks
punctuation_count = len([char for char in text if char in string.punctuation])

# Remove stop words
words_no_stopwords = [word for word in words_cleaned if word.lower() not in stop_words_ukr]

# Count statistical information
initial_word_count = len(words_cleaned)
stop_word_count = len([word for word in words_cleaned if word.lower() in stop_words_ukr])
final_word_count = len(words_no_stopwords)

# Initialize stemmer and lemmatizer
stemmer = SnowballStemmer("russian")  # Using Russian stemmer as a substitute for Ukrainian
lemmatizer = WordNetLemmatizer()

# Perform stemming
stemmed_words = [stemmer.stem(word) for word in words_no_stopwords]

# Perform lemmatization
lemmatized_words = [lemmatizer.lemmatize(word) for word in words_no_stopwords]

# Output the results
print(f"Initial text volume: {initial_word_count} words")
print(f"Number of punctuation marks: {punctuation_count}")
print(f"Number of stop words: {stop_word_count}")
print(f"Word count in final text: {final_word_count}")

print("\nStemming results:")
print(stemmed_words)

print("\nLemmatization results:")
print(lemmatized_words)

# Comparison of original, stemmed, and lemmatized words
comparison = list(zip(words_no_stopwords, stemmed_words, lemmatized_words))
print("\nComparison of original words, stemming, and lemmatization:")
for original, stemmed, lemmatized in comparison:
    print(f"Original: {original}, Stemmed: {stemmed}, Lemmatized: {lemmatized}")

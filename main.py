import whisper
import re
import json
from difflib import SequenceMatcher

# take input from the user
num = int(input("Enter a surah number: "))


# files are named as 001.mp3, 002.mp3, 003.mp3, ...
# so we need to add the leading zeros
input_file = "Quran_recitations/" + str(num).zfill(3) + ".mp3"

# check if the file exists
try:
    with open(input_file):
        pass
except FileNotFoundError:
    print("File not found")

# Load model and move to GPU
model = whisper.load_model("medium").to("cuda")

whisper_output = model.transcribe(input_file, language="ar", word_timestamps=True)

def normalize_arabic(text):
    # Remove leading/trailing whitespace
    text = text.strip()
    # Remove optional leading numbers with punctuation and spaces
    text = re.sub(r'^\s*[0-9]+[.:،]?\s*', '', text)
    # Remove tatweel and diacritics
    text = re.sub(r'[\u0640]', '', text)
    text = re.sub(r'[\u064b-\u065f]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}".replace('.', ',')

# === Load data ===
# Quran JSON (format: { "78:1": "text", "78:2": "text", ... } for surah Naba, for example)
with open('quran.json', encoding='utf-8') as f:
    quran = json.load(f)
with open('translation.json', encoding='utf-8') as f:
    translations = json.load(f)

# Assume whisper_output is defined and contains word-level timestamps.
# Example: whisper_output['segments'] is a list of segments; each segment has a "words" list.
whisper_words = []
for segment in whisper_output['segments']:
    for word_info in segment['words']:
        word = normalize_arabic(word_info['word'])
        if word:  # skip empty tokens
            whisper_words.append({
                'text': word,
                'start': word_info['start'],
                'end': word_info['end']
            })

# === Build the expected word sequence from Quran ===
surah = str(num)
filtered_verses = sorted([vk for vk in quran.keys() if vk.startswith(surah + ":")],
                          key=lambda x: int(x.split(':')[1]))
expected_words = []    # list of words for the entire surah
verse_boundaries = {}  # map verse key to a tuple (start_index, end_index) in expected_words

for vk in filtered_verses:
    verse_text = normalize_arabic(quran[vk])
    words = verse_text.split()
    start_idx = len(expected_words)
    expected_words.extend(words)
    end_idx = len(expected_words) - 1
    verse_boundaries[vk] = (start_idx, end_idx)

# === Build the transcribed word sequence from Whisper ===
transcribed_words = [w['text'] for w in whisper_words]

# === Global alignment using SequenceMatcher ===
matcher = SequenceMatcher(None, expected_words, transcribed_words)
matching_blocks = matcher.get_matching_blocks()
# matching_blocks is a list of triples (i, j, n)
# where expected_words[i:i+n] == transcribed_words[j:j+n]

# We'll build a mapping from each expected word index to a transcribed word index.
# For indices that are matched, we record the transcribed index.
mapping = {}
for block in matching_blocks:
    for offset in range(block.size):
        mapping[block.a + offset] = block.b + offset

# === Now, for each verse, determine the corresponding timestamps ===
subtitles = []
for vk, (exp_start, exp_end) in verse_boundaries.items():
    # We want to find the earliest and latest transcribed indices corresponding to this verse.
    trans_indices = [mapping[i] for i in range(exp_start, exp_end + 1) if i in mapping]
    if not trans_indices:
        print(f"Warning: No alignment for verse {vk}")
        continue
    t_start = min(trans_indices)
    t_end = max(trans_indices)
    # Get timestamps from whisper_words
    if t_start < len(whisper_words) and t_end < len(whisper_words):
        start_time = whisper_words[t_start]['start']
        end_time = whisper_words[t_end]['end']
        subtitles.append({
            'verse': vk,
            'start': start_time,
            'end': end_time,
            'translation': translations.get(vk, "Translation not found")
        })
    else:
        print(f"Index error for verse {vk}")

# === Generate SRT file ===
srt_lines = []
for idx, sub in enumerate(subtitles, 1):
    start = to_srt_time(sub['start'])
    end = to_srt_time(sub['end'])
    srt_lines.append(f"{idx}\n{start} --> {end}\n{sub['translation']}\n")

# For example, if input_file is defined (like "078.mp3"), then:
output_file = f"subtitles/{input_file.split('/')[-1].split('.')[0]}.srt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(srt_lines))

print(f"SRT file generated successfully: {output_file}")

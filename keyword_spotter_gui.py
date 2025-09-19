
import os
import json
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
import streamlit as st

# کلاس‌های موجود از کد شما
class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def pre_emphasis(self, signal, pre_emph_coef=0.97):
        return np.append(signal[0], signal[1:] - pre_emph_coef * signal[:-1])

    def remove_silence(self, y, threshold=0.01):
        try:
            energy = librosa.feature.rms(y=y)[0]
            frames = np.nonzero(energy > threshold)
            if frames[0].size:
                y = y[librosa.frames_to_samples(frames[0][0]):librosa.frames_to_samples(frames[0][-1])]
            return y
        except Exception as e:
            print(f"Error in remove_silence: {e}")
            return y

    def extract_mfcc(self, y):
        try:
            y = self.pre_emphasis(y)
            mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc,
                                        n_fft=self.n_fft, hop_length=self.hop_length,
                                        n_mels=self.n_mels)
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
            return mfcc
        except Exception as e:
            print(f"Error in extract_mfcc: {e}")
            return None

    def extract_delta_features(self, mfcc):
        try:
            delta = librosa.feature.delta(mfcc)
            delta_delta = librosa.feature.delta(mfcc, order=2)
            combined = np.vstack([mfcc, delta, delta_delta])
            return combined
        except Exception as e:
            print(f"Error in extract_delta_features: {e}")
            return None

    def normalize_feature_length(self, features, target_length=50):
        try:
            n_features, original_length = features.shape
            x_old = np.linspace(0, 1, original_length)
            x_new = np.linspace(0, 1, target_length)
            interpolated = np.zeros((n_features, target_length))
            for i in range(n_features):
                f = interp1d(x_old, features[i], kind='linear')
                interpolated[i] = f(x_new)
            return interpolated
        except Exception as e:
            print(f"Error in normalize_feature_length: {e}")
            return None

    def extract_energy(self, y):
        try:
            return np.sum(y ** 2)
        except Exception as e:
            print(f"Error in extract_energy: {e}")
            return 0.0

    def save_mfcc_spectrogram(self, mfcc, title, energy, plot_dir):
        try:
            plt.figure(figsize=(8, 3))
            librosa.display.specshow(mfcc, x_axis='time', sr=self.sample_rate)
            plt.colorbar()
            plt.title(f'MFCC: {title}')
            text_str = f'Energy: {energy:.2f}'
            plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.tight_layout()
            output_path = os.path.join(plot_dir, f'mfcc_{title}.png')
            plt.savefig(output_path)
            plt.close()
            print(f"✅ Saved MFCC spectrogram: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error in save_mfcc_spectrogram for {title}: {e}")
            return None

class SubsequenceDTWKeywordDetector:
    def __init__(self, keyword_feature_dir='data/mfcc_features', 
                 speech_feature_dir='data/mfcc_features_continuous_speech', 
                 annotation_dir='data/annotations', 
                 output_dir='data/optimized_keyword_detections',
                 target_length=50, sample_rate=16000, hop_length=512):
        self.keyword_feature_dir = keyword_feature_dir
        self.speech_feature_dir = speech_feature_dir
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        os.makedirs(self.output_dir, exist_ok=True)
        self.keyword_map = {
            'Salam': 'سلام',
            'Bale': 'بله',
            'Kheir': 'خیر',
            'Motshakeram': 'متشکرم',
            'BesiarAali': 'بسیار عالی',
            'Khodahafez': 'خداحافظ'  # Added Khodahafez
        }

    def compute_subsequence_dtw(self, query, sequence, distance_metric='euclidean'):
        n, m = query.shape[1], sequence.shape[1]
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, :] = 0
        predecessors = np.zeros((n + 1, m + 1, 2), dtype=int)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if distance_metric == 'euclidean':
                    cost = euclidean(query[:, i-1], sequence[:, j-1])
                else:
                    raise ValueError("Unsupported distance metric")
                costs = [
                    cost_matrix[i-1, j-1],  # Match
                    cost_matrix[i-1, j],    # Insertion
                    cost_matrix[i, j-1]     # Deletion
                ]
                min_cost_idx = np.argmin(costs)
                cost_matrix[i, j] = cost + costs[min_cost_idx]
                predecessors[i, j] = [(i-1, j-1), (i-1, j), (i, j-1)][min_cost_idx]

        end_idx = np.argmin(cost_matrix[n, 1:]) + 1
        min_cost = cost_matrix[n, end_idx]
        path = []
        i, j = n, end_idx
        while i > 0:
            path.append((i-1, j-1))
            i, j = predecessors[i, j]
        path.reverse()
        start_idx = path[0][1] if path else 0
        return min_cost, start_idx, end_idx

    def frame_to_time(self, frame_idx):
        return frame_idx * self.hop_length / self.sample_rate

    def detect_keywords(self, speech_file, keywords, threshold=40.0):
        speech_path = os.path.join(self.speech_feature_dir, speech_file)
        try:
            speech_mfcc = np.load(speech_path)
            print(f"Loaded speech MFCC: {speech_file}, Shape: {speech_mfcc.shape}")
        except FileNotFoundError:
            print(f"Error: Speech file {speech_path} not found.")
            return None

        if speech_mfcc.shape[0] != 39:
            print(f"Error: Unexpected shape for {speech_file}: {speech_mfcc.shape}. Expected (39, N).")
            return None

        annotation_path = os.path.join(self.annotation_dir, speech_file.replace('.npy', '.json'))
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            duration = annotation.get('duration', 0.0)
            text = annotation.get('text', '')
            print(f"Annotation loaded: {text}, Duration: {duration}")
        except FileNotFoundError:
            print(f"Warning: Annotation file {annotation_path} not found. Using default values.")
            duration = speech_mfcc.shape[1] * self.hop_length / self.sample_rate
            text = 'Unknown'

        allowed_keywords = []
        for keyword in keywords:
            persian_keyword = self.keyword_map.get(keyword, keyword)
            if persian_keyword in text:
                allowed_keywords.append(keyword)
        print(f"Allowed keywords for {speech_file}: {allowed_keywords}")

        detections = {}
        for keyword in allowed_keywords:
            keyword_files = [f for f in os.listdir(self.keyword_feature_dir) 
                           if f.startswith(keyword + '_') and f.endswith('.npy')]
            if not keyword_files:
                print(f"Warning: No feature files found for keyword {keyword}.")
                continue

            best_distance = float('inf')
            best_detection = None
            for keyword_file in keyword_files:
                keyword_path = os.path.join(self.keyword_feature_dir, keyword_file)
                try:
                    keyword_mfcc = np.load(keyword_path)
                    print(f"Loaded keyword MFCC: {keyword_file}, Shape: {keyword_mfcc.shape}")
                except FileNotFoundError:
                    print(f"Error: Keyword file {keyword_path} not found.")
                    continue

                if keyword_mfcc.shape != (39, self.target_length):
                    print(f"Error: Unexpected shape for {keyword_file}: {keyword_mfcc.shape}. Expected (39, {self.target_length}).")
                    continue

                distance, start_idx, end_idx = self.compute_subsequence_dtw(keyword_mfcc, speech_mfcc)
                start_time = self.frame_to_time(start_idx)
                end_time = self.frame_to_time(end_idx)
                print(f"Keyword: {keyword_file}, DTW Distance: {distance:.2f}, Start Time: {start_time:.2f}s, End Time: {end_time:.2f}s")

                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_detection = {
                        'word': self.keyword_map.get(keyword, keyword),
                        'start_time': round(start_time, 2),
                        'end_time': round(end_time, 2),
                        'distance': distance  # برای محاسبه confidence score
                    }

            if best_detection:
                detections[keyword] = best_detection

        output_detections = [det for det in detections.values() if det]
        output_detections.sort(key=lambda x: x['start_time'])

        output = {
            'filename': speech_file.replace('.npy', '.wav'),
            'text': text,
            'duration': round(duration, 6),
            'keywords': output_detections
        }

        output_path = os.path.join(self.output_dir, speech_file.replace('.npy', '.json'))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f"✅ Saved detections to {output_path}")
        return output

# توابع پیش‌پردازش صوتی
def normalize_audio(y):
    return y / np.max(np.abs(y))

def trim_silence(y, top_db=30):
    return librosa.effects.trim(y, top_db=top_db)[0]

def bandpass_filter(y, sr, low_cut=300, high_cut=3400, order=5):
    nyquist = 0.5 * sr
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, y)

class KeywordSpotterGUI:
    def __init__(self, spotter):
        self.spotter = spotter
        self.extractor = FeatureExtractor()
        self.input_dir = 'data/Sentence'
        self.preprocessed_dir = 'data/continuous_speech'
        self.mfcc_dir = 'data/mfcc_features_continuous_speech'
        self.plot_dir = 'data/optimized_keyword_plots'
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        os.makedirs(self.mfcc_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def preprocess_audio(self, input_path, filename):
        try:
            y, sr = librosa.load(input_path, sr=16000)
            y = normalize_audio(y)
            y = trim_silence(y)
            y = bandpass_filter(y, sr, low_cut=300, high_cut=3400, order=5)
            output_path = os.path.join(self.preprocessed_dir, filename)
            sf.write(output_path, y, sr, subtype='PCM_16')
            print(f"✅ Preprocessed and saved: {output_path}")
            return y, sr
        except Exception as e:
            print(f"Error in preprocess_audio for {filename}: {e}")
            return None, None

    def extract_features(self, y, filename):
        try:
            y = self.extractor.remove_silence(y)
            if y is None or len(y) == 0:
                print(f"Warning: Empty audio after silence removal for {filename}")
                return None
            energy = self.extractor.extract_energy(y)
            mfcc = self.extractor.extract_mfcc(y)
            if mfcc is None:
                print(f"Warning: Failed to extract MFCC for {filename}")
                return None
            mfcc = self.extractor.extract_delta_features(mfcc)
            if mfcc is None:
                print(f"Warning: Failed to extract delta features for {filename}")
                return None
            mfcc = self.extractor.normalize_feature_length(mfcc)
            if mfcc is None:
                print(f"Warning: Failed to normalize feature length for {filename}")
                return None
            npy_filename = filename.replace('.wav', '.npy')
            npy_path = os.path.join(self.mfcc_dir, npy_filename)
            np.save(npy_path, mfcc)
            print(f"✅ Saved MFCC features: {npy_path}")
            mfcc_plot_path = self.extractor.save_mfcc_spectrogram(mfcc, filename, energy, self.plot_dir)
            return mfcc, npy_filename, mfcc_plot_path
        except Exception as e:
            print(f"Error in extract_features for {filename}: {e}")
            return None, None, None

    def visualize_results(self, audio_data, sample_rate, detections, filename):
        try:
            plt.figure(figsize=(15, 6))
            librosa.display.waveshow(audio_data, sr=sample_rate)
            plt.title(f'Waveform: {filename}')
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")

            # محاسبه confidence score (معکوس فاصله DTW نرمال‌شده)
            max_distance = 40.0  # آستانه DTW
            for detection in detections.get('keywords', []):
                start_time = detection['start_time']
                end_time = detection['end_time']
                word = detection['word']
                distance = detection.get('distance', max_distance)
                confidence = max(0, 1 - distance / max_distance)  # Confidence بین 0 و 1
                # رسم محدوده کلمه کلیدی
                plt.axvspan(start_time, end_time, alpha=0.3, color='green', label=word)
                plt.text(start_time, 0.8, f'{word} ({confidence:.2f})', fontsize=10, color='black',
                         bbox=dict(facecolor='white', alpha=0.8))

            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(self.plot_dir, f'waveform_detections_{filename}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"✅ Saved waveform plot with detections: {plot_path}")
            return plot_path
        except Exception as e:
            print(f"Error in visualize_results for {filename}: {e}")
            return None

    def create_interactive_demo(self):
        st.title("Keyword Spotting Demo")
        st.write("Upload a WAV audio file to detect keywords.")
        
        uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
        
        if uploaded_file is not None:
            filename = uploaded_file.name
            input_path = os.path.join(self.input_dir, filename)
            # ذخیره فایل موقت
            os.makedirs(self.input_dir, exist_ok=True)
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.write(f"Processing file: {filename}")
            
            # پیش‌پردازش صوتی
            y, sr = self.preprocess_audio(input_path, filename)
            if y is None:
                st.error(f"Failed to preprocess {filename}")
                return
            
            # نمایش شکل موج اولیه
            st.write("Original Waveform")
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr)
            st.pyplot(plt)
            plt.close()
            
            # استخراج ویژگی‌ها
            mfcc, npy_filename, mfcc_plot_path = self.extract_features(y, filename)
            if mfcc is None:
                st.error(f"Failed to extract features for {filename}")
                return
            
            # نمایش نمودار MFCC
            if mfcc_plot_path:
                st.write("MFCC Spectrogram")
                st.image(mfcc_plot_path)
            
            # تشخیص کلمات کلیدی
            keywords = ['Salam', 'Bale', 'Kheir', 'Motshakeram', 'BesiarAali', 'Khodahafez']  # Added 'Khodahafez'
            detections = self.spotter.detect_keywords(npy_filename, keywords, threshold=40.0)
            if detections is None:
                st.error(f"Failed to detect keywords for {filename}")
                return
            
            # نمایش نتایج تشخیص
            st.write("Detected Keywords:")
            st.json(detections)
            
            # نمایش شکل موج با محدوده‌های کلمات کلیدی
            waveform_plot_path = self.visualize_results(y, sr, detections, filename)
            if waveform_plot_path:
                st.write("Waveform with Detected Keywords")
                st.image(waveform_plot_path)

def main():
    spotter = SubsequenceDTWKeywordDetector()
    gui = KeywordSpotterGUI(spotter)
    gui.create_interactive_demo()

if __name__ == '__main__':
    main()

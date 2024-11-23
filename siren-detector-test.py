import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import tkinter as tk
from threading import Thread, Event
import queue
import time
import csv
import urllib.request
import os


class SirenDetector:
    def __init__(self):
        # Sınıf isimleri için CSV dosyasını indir
        self.class_map_path = 'yamnet_class_map.csv'
        if not os.path.exists(self.class_map_path):
            print("Sınıf haritası indiriliyor...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv',
                self.class_map_path)

        # Sınıf isimlerini yükle
        self.class_names = []
        with open(self.class_map_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.class_names.append(row['display_name'])

        # YAMNet modelini yükle
        print("Model yükleniyor...")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("Model yüklendi")

        # Ses parametreleri
        self.CHANNELS = 1
        self.RATE = 16000  # YAMNet için sabit
        self.CHUNK = int(self.RATE * 1)  # 1 saniyelik chunk
        self.WINDOW_SIZE = 1  # 1 saniyelik pencere
        self.prediction_cooldown = 0.2  # 200ms'de bir analiz yap

        # İlgili sınıflar
        self.target_classes = [
            'Emergency vehicle', 'Siren', 'Police car (siren)',
            'Ambulance (siren)', 'Fire engine, fire truck (siren)'
        ]

        # İşlem parametreleri
        self.buffer = []
        self.is_recording = Event()
        self.audio_queue = queue.Queue()
        self.last_prediction_time = 0
        self.clean_exit = False

        # Ardışık tespit için sayaç
        self.emergency_counter = 0
        self.MIN_DETECTIONS = 2  # En az kaç kez tespit edilmeli

        # GUI oluştur
        self.create_gui()

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Acil Durum Sesi Algılayıcı")
        self.root.geometry("400x600")

        # Ana frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')

        # Başlık
        title_label = tk.Label(main_frame,
                               text="Acil Durum Sesi\nAlgılayıcı",
                               font=('Arial', 24, 'bold'),
                               bg='#f0f0f0',
                               fg='#2c3e50')
        title_label.pack(pady=20)

        # Durum etiketi
        self.status_label = tk.Label(main_frame,
                                     text="Bekleniyor...",
                                     font=('Arial', 14),
                                     bg='#f0f0f0')
        self.status_label.pack(pady=10)

        # Ses seviyesi göstergesi
        self.level_label = tk.Label(main_frame,
                                    text="Ses Seviyesi: -",
                                    font=('Arial', 12),
                                    bg='#f0f0f0')
        self.level_label.pack(pady=10)

        # Debug etiketi (Anlık tespitler)
        self.debug_label = tk.Label(main_frame,
                                    text="",
                                    font=('Arial', 10),
                                    bg='#f0f0f0',
                                    justify=tk.LEFT)
        self.debug_label.pack(pady=10)

        # Sonuç etiketi
        self.result_label = tk.Label(main_frame,
                                     text="",
                                     font=('Arial', 18, 'bold'),
                                     bg='#f0f0f0',
                                     wraplength=350)
        self.result_label.pack(pady=20)

        # Butonlar için frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=20)

        # Başlat/Durdur butonu
        self.toggle_button = tk.Button(button_frame,
                                       text="Başlat",
                                       command=self.toggle_recording,
                                       width=20,
                                       height=2,
                                       font=('Arial', 12))
        self.toggle_button.pack(pady=10)

        # Çıkış butonu
        self.quit_button = tk.Button(button_frame,
                                     text="Çıkış",
                                     command=self.quit_app,
                                     width=20,
                                     height=2,
                                     font=('Arial', 12))
        self.quit_button.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if not self.clean_exit:
            self.audio_queue.put(indata[:, 0])

    def process_audio(self):
        print("Ses işleme başladı")

        while self.is_recording.is_set():
            if not self.audio_queue.empty():
                new_data = self.audio_queue.get()
                self.buffer.extend(new_data)

                max_buffer_size = int(self.RATE * self.WINDOW_SIZE)
                if len(self.buffer) > max_buffer_size:
                    self.buffer = self.buffer[-max_buffer_size:]

                current_time = time.time()
                if len(self.buffer) >= self.CHUNK and \
                        current_time - self.last_prediction_time > self.prediction_cooldown:

                    audio_data = np.array(self.buffer[-self.CHUNK:])

                    # Ses seviyesini güncelle
                    rms = np.sqrt(np.mean(np.square(audio_data)))
                    db = 20 * np.log10(rms + 1e-10)
                    self.level_label.config(text=f"Ses Seviyesi: {db:.1f} dB")

                    # YAMNet'e gönder
                    scores, embeddings, spectrogram = self.model(audio_data)
                    scores = scores.numpy()

                    # En yüksek skorlu 5 tahmini al
                    top_indices = np.argsort(scores.mean(axis=0))[-5:][::-1]

                    # Debug bilgisi güncelle
                    debug_text = "Anlık Tespitler:\n"
                    for idx in top_indices:
                        score = scores.mean(axis=0)[idx]
                        if score > 0.1:  # Sadece belirli bir eşiğin üstündeki tahminleri göster
                            debug_text += f"{self.class_names[idx]}: {score:.2f}\n"
                    self.debug_label.config(text=debug_text)

                    # Acil durum aracı sesi kontrolü
                    emergency_detected = False
                    max_score = 0
                    detected_class = ""

                    for idx in top_indices:
                        class_name = self.class_names[idx]
                        score = scores.mean(axis=0)[idx]

                        if class_name in self.target_classes and score > max_score:
                            emergency_detected = True
                            max_score = score
                            detected_class = class_name

                    # Tespit sayacını güncelle
                    if emergency_detected and max_score > 0.3:
                        self.emergency_counter += 1
                    else:
                        self.emergency_counter = max(0, self.emergency_counter - 1)

                    # Sonucu göster
                    if self.emergency_counter >= self.MIN_DETECTIONS:
                        self.result_label.config(
                            text=f"!!! ACİL DURUM !!!\nTespit: {detected_class}\nGüven: {max_score:.2f}",
                            fg="red"
                        )
                        self.status_label.config(
                            text="DİKKAT: Acil Durum Aracı!",
                            fg="red"
                        )
                    else:
                        self.result_label.config(text="")
                        self.status_label.config(
                            text="Dinleniyor...",
                            fg="green"
                        )

                    self.last_prediction_time = current_time

    def toggle_recording(self):
        if not self.is_recording.is_set():
            print("Kayıt başlatılıyor...")
            self.is_recording.set()
            self.buffer = []
            self.emergency_counter = 0  # Sayacı sıfırla
            self.toggle_button.config(text="Durdur")
            self.status_label.config(text="Dinleniyor...", fg="green")

            try:
                self.stream = sd.InputStream(
                    channels=self.CHANNELS,
                    samplerate=self.RATE,
                    callback=self.audio_callback,
                    blocksize=int(self.RATE * 0.1)  # 100ms'lik bloklar
                )
                self.stream.start()
                print("Ses akışı başlatıldı")

                self.process_thread = Thread(target=self.process_audio)
                self.process_thread.start()
                print("İşleme thread'i başlatıldı")

            except Exception as e:
                print(f"Ses akışı başlatma hatası: {str(e)}")
                self.is_recording.clear()
                self.toggle_button.config(text="Başlat")
                self.status_label.config(text="Hata!", fg="red")

        else:
            print("Kayıt durduruluyor...")
            self.is_recording.clear()
            self.toggle_button.config(text="Başlat")
            self.status_label.config(text="Bekleniyor...", fg="black")
            self.result_label.config(text="")

            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
                print("Ses akışı durduruldu")

    def quit_app(self):
        print("Uygulama kapatılıyor...")
        self.clean_exit = True
        self.is_recording.clear()

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        if hasattr(self, 'process_thread'):
            if self.process_thread.is_alive():
                self.process_thread.join(timeout=1.0)

        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    detector = SirenDetector()
    detector.root.mainloop()
import os
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt  # <-- AJOUTER CETTE LIGNE
from PyQt5.QtGui import QPixmap, QImage
import cv2
import sys
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

# Configuration
plt.switch_backend('Agg')

qtcreator_file = "design.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(DesignWindow, self).__init__()
        self.setupUi(self)

        # Variables audio
        self.audio_fs = None
        self.audio_signal = None
        self.audio_original = None

        # Variables vidéo
        self.video_path = None
        self.video_fps = None
        self.video_width = None
        self.video_height = None
        self.video_frame_count = None
        self.video_size = None

        # Connexion des boutons
        self.pushButton.clicked.connect(self.handle_load_audio)
        self.pushButton_2.clicked.connect(self.handle_resampling)
        self.pushButton_3.clicked.connect(self.handle_audio_compression)
        self.pushButton_4.clicked.connect(self.handle_load_video)
        self.pushButton_5.clicked.connect(self.handle_video_compression)

        # Boutons radio
        self.radioButton.setChecked(True)

        # Liste des codecs
        self.listWidget.addItems(["mp4v", "MJPG", "XVID"])

        # Dossier ressources
        if not os.path.exists("ressources"):
            os.makedirs("ressources")

    def get_audio_info(self, fs, signal):
        if signal.ndim == 1:
            n, c = len(signal), 1
            type_audio = "Mono"
        else:
            n, c = signal.shape
            type_audio = "Stéréo"
        duree = n / fs
        return {'type': type_audio, 'fs': fs, 'n_echantillons': n, 'duree': duree, 'canaux': c}

    def plot_to_pixmap(self, signal, fs, title=""):
        plt.ioff()
        plt.clf()
        duree = len(signal) / fs
        t = np.linspace(0, duree, min(len(signal), 3000))
        if len(signal) > 3000:
            signal = signal[:3000]
            t = t[:3000]
        plt.figure(figsize=(10, 3))
        plt.plot(t, signal, color='blue', linewidth=0.8)
        plt.xlabel("Temps (s)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("temp_plot.png")
        plt.close()
        return QPixmap("temp_plot.png")

    def plot_comparison(self, original, resampled, factor):
        plt.ioff()
        plt.clf()
        plt.figure(figsize=(10, 3))
        n_display = min(3000, len(original))
        plt.plot(original[:n_display], 'blue', linewidth=0.8, label="Original")
        n_resampled = min(3000 // factor, len(resampled))
        indices = np.arange(0, n_resampled) * factor
        plt.plot(indices, resampled[:n_resampled], 'red', linewidth=0.6, label=f"Fe/{factor}")
        plt.xlabel("Échantillons")
        plt.ylabel("Amplitude")
        plt.title(f"Comparaison (facteur {factor})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("temp_compare.png")
        plt.close()
        return QPixmap("temp_compare.png")

    def plot_spectrum(self, signal, fs):
        plt.ioff()
        plt.clf()
        N = len(signal)
        z = fft(signal)
        modules = np.abs(z)
        f = fftfreq(N, 1 / fs)
        N2 = min(N // 2, 5000)
        plt.figure(figsize=(10, 3))
        plt.plot(f[:N2], modules[:N2], 'green', linewidth=0.8)
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Spectre du signal")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("temp_spectrum.png")
        plt.close()
        return QPixmap("temp_spectrum.png")

    def handle_load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Charger audio", "ressources", "WAV (*.wav)")
        if not file_path:
            return
        try:
            self.audio_fs, self.audio_signal = wavfile.read(file_path)
            info = self.get_audio_info(self.audio_fs, self.audio_signal)
            text = f"Type: {info['type']}\nFréquence: {info['fs']} Hz\nÉchantillons: {info['n_echantillons']}\nDurée: {info['duree']:.2f} s"
            self.label_4.setText(text)
            if self.audio_signal.ndim > 1:
                signal = self.audio_signal[:, 0]
            else:
                signal = self.audio_signal
            pix = self.plot_to_pixmap(signal, self.audio_fs, "Signal original")
            self.label.setPixmap(pix)
            self.label_2.clear()
            self.label_3.clear()
            QMessageBox.information(self, "Succès", f"Audio chargé! Durée: {info['duree']:.2f}s")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def handle_resampling(self):
        if self.audio_signal is None:
            QMessageBox.warning(self, "Attention", "Chargez d'abord un audio!")
            return
        if self.radioButton.isChecked():
            factor = 2
        elif self.radioButton_2.isChecked():
            factor = 4
        else:
            factor = 8
        try:
            if self.audio_signal.ndim > 1:
                signal = self.audio_signal[:, 0]
            else:
                signal = self.audio_signal
            resampled = signal[::factor]
            pix = self.plot_comparison(signal, resampled, factor)
            self.label_2.setPixmap(pix)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def handle_audio_compression(self):
        if self.audio_signal is None:
            QMessageBox.warning(self, "Attention", "Chargez d'abord un audio!")
            return
        try:
            if self.audio_signal.ndim > 1:
                signal = self.audio_signal[:, 0]
            else:
                signal = self.audio_signal
            z = fft(signal)
            modules = np.abs(z)
            modules_tries = np.sort(modules)
            N = len(z)
            indice_seuil = int(N * (1 - 1 / 128))
            seuil = modules_tries[indice_seuil]
            z_comp = z.copy()
            z_comp[modules < seuil] = 0
            signal_comp = np.real(ifft(z_comp))
            pix = self.plot_spectrum(signal_comp, self.audio_fs)
            self.label_3.setPixmap(pix)
            QMessageBox.information(self, "Succès", "Compression terminée!")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def handle_load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Charger vidéo", "ressources", "Vidéo (*.avi *.mp4)")
        if not file_path:
            return
        try:
            self.video_path = file_path
            cap = cv2.VideoCapture(file_path)
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_size = os.path.getsize(file_path) / (1024 * 1024)
            text = f"Résolution: {self.video_width}x{self.video_height}\nFPS: {self.video_fps:.2f}\nTrames: {self.video_frame_count}\nDurée: {self.video_frame_count / self.video_fps:.2f}s\nTaille: {self.video_size:.2f} Mo"
            self.label_6.setText(text)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qt_img)
                self.label_5.setPixmap(pix.scaled(self.label_5.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            cap.release()
            QMessageBox.information(self, "Succès", "Vidéo chargée!")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def handle_video_compression(self):
        if self.video_path is None:
            QMessageBox.warning(self, "Attention", "Chargez d'abord une vidéo!")
            return
        try:
            try:
                new_fps = float(self.textEdit.toPlainText()) if self.textEdit.toPlainText() else self.video_fps
            except:
                new_fps = self.video_fps
            try:
                new_w = int(self.textEdit_2.toPlainText()) if self.textEdit_2.toPlainText() else self.video_width
            except:
                new_w = self.video_width
            try:
                new_h = int(self.textEdit_3.toPlainText()) if self.textEdit_3.toPlainText() else self.video_height
            except:
                new_h = self.video_height
            codec = self.listWidget.currentItem().text() if self.listWidget.currentItem() else "mp4v"
            fourcc_map = {"mp4v": cv2.VideoWriter_fourcc(*'mp4v'), "MJPG": cv2.VideoWriter_fourcc(*'MJPG'),
                          "XVID": cv2.VideoWriter_fourcc(*'XVID')}
            fourcc = fourcc_map.get(codec, cv2.VideoWriter_fourcc(*'mp4v'))
            cap = cv2.VideoCapture(self.video_path)
            out = cv2.VideoWriter("ressources/compressed.avi", fourcc, new_fps, (new_w, new_h))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, (new_w, new_h))
                out.write(frame_resized)
                QtWidgets.QApplication.processEvents()
            cap.release()
            out.release()
            new_size = os.path.getsize("ressources/compressed.avi") / (1024 * 1024)
            economy = ((self.video_size - new_size) / self.video_size) * 100
            text = f"Codec: {codec}\nDimensions: {new_w}x{new_h}\nFPS: {new_fps}\n\nTaille originale: {self.video_size:.2f} Mo\nNouvelle taille: {new_size:.2f} Mo\nÉconomie: {economy:.1f}%"
            self.label_7.setText(text)
            QMessageBox.information(self, "Succès", f"Compression terminée! Économie: {economy:.1f}%")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())
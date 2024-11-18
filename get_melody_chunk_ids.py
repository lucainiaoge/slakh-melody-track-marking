import os
import json
import argparse
import numpy as np
import pretty_midi
import pickle

from midi_features import get_track_features

PIANO_ROLL_SAMPLE_RATE = 100
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in midi melody track classification')
    parser.add_argument(
        '-slakh-dir', type=str,
        help='The folder of slakh dataset'
    )
    parser.add_argument(
        '-model-path', type=str, default="classifier_model.pkl",
        help='The path to classifier sklearn model pipeline (pickle format)'
    )
    parser.add_argument(
        '--chunk-sec', type=float, default=10.0,
        help='The chunk duration in processing slakh dataset'
    )
    parser.add_argument(
        '--melody-threshold', type=float, default=0.2,
        help='The min energy percentage that a chunk should be classified as a melody chunk'
    )
    args = parser.parse_args()
    
    with open(args.model_path, 'rb') as f:
        pipeline = pickle.load(f)

    chunk_len = int(args.chunk_sec * PIANO_ROLL_SAMPLE_RATE)
    slakh_pieces = os.listdir(args.slakh_dir)
    for piece_folder in slakh_pieces:
        track_dir = os.path.join(args.slakh_dir, piece_folder)
        if (not os.path.isdir(track_dir)) or ("Track" not in str(piece_folder)):
            continue
        
        midi_path = os.path.join(track_dir, "all_src.mid")
        pretty_midi_features = pretty_midi.PrettyMIDI(midi_path)
        track_features = get_track_features(pretty_midi_features)
        
        y_pred = pipeline.predict_proba(track_features)[:,1]
        channel_pred = np.argmax(y_pred)
        print(f"Predicted melody channel for {piece_folder}: {channel_pred}")

        melody_piano_roll = pretty_midi_features.instruments[channel_pred].get_piano_roll(fs=PIANO_ROLL_SAMPLE_RATE)
        melody_energy_vec = melody_piano_roll.sum(axis = 0)
        average_energy = melody_energy_vec[melody_energy_vec > 0].mean()
        min_energy = average_energy * args.melody_threshold
        
        num_chunks = int(len(melody_energy_vec) / chunk_len)
        melody_chunk_ids = []
        for i_chunk in range(num_chunks):
            chunk_start = int(i_chunk * chunk_len)
            this_chunk_vec = melody_energy_vec[chunk_start:chunk_start+chunk_len]
            chunk_energy = this_chunk_vec.mean()
            if chunk_energy > min_energy:
                melody_chunk_ids.append(i_chunk)
                
        melody_chunk_json_filename = f"melody_chunks_ids_with_interval_{int(args.chunk_sec)}_sec.json"
        melody_chunk_json_path = os.path.join(track_dir, melody_chunk_json_filename)
        with open(melody_chunk_json_path, "w") as f:
        	json.dump(melody_chunk_ids, f)

        print(f"Assigned melody chunk ids: {melody_chunk_ids}; saved to json")
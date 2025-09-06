# def spec_augment(mel_spectrogram, config: SpecAugmentConfig):
#     """ Applies SpecAugment. """
#     mel_spectrogram_aug = jnp.array(mel_spectrogram)
    
#     # Freq masking
#     for _ in range(config.n_freq_masks):
#         f = np.random.uniform(low=0.0, high=config.freq_mask_param)
#         f = int(f)
#         f0 = np.random.randint(0, mel_spectrogram_aug.shape[1] - f)
#         mel_spectrogram_aug = mel_spectrogram_aug.at[:, f0:f0+f].set(0)

#     # Time masking
#     for _ in range(config.n_time_masks):
#         t = np.random.uniform(low=0.0, high=config.time_mask_param)
#         t = int(t)
#         t0 = np.random.randint(0, mel_spectrogram_aug.shape[0] - t)
#         mel_spectrogram_aug = mel_spectrogram_aug.at[t0:t0+t, :].set(0)

#     return mel_spectrogram_aug
"""Signal-processing chain: windows, FFTs, MTI, CFAR, clustering, DoA, point
clouds, micro-Doppler."""
from .windows import get_window, coherent_gain, processing_loss_db, enbw_bins
from .fft import (range_fft, doppler_fft, mti, rd_power, to_db, range_doppler,
                  bin_to_range, bin_to_velocity)
from .cfar import (cfar_2d, cfar_1d, CfarResult, ca_alpha, os_alpha, gs_alpha, cfar_alpha,
                   os_rank_index, os_pfa, go_pfa_beta, so_pfa_beta, gs_pfa_beta)
from .detect import (parabolic_offset, refine_peak, refine_peak_1d, Cluster, cluster_connected,
                     cluster_dbscan, local_maxima)
from .doa import (tdm_phase, tdm_compensate, steering_vector, azimuth_ula, UlaSelection, angle_fft,
                  angle_bin_to_sin, angle_bin_to_deg, sin_to_angle_bin, covariance, mvdr_spectrum,
                  music_spectrum, bartlett_spectrum, doa_spectrum, find_peaks_1d,
                  estimate_elevation, range_angle_map)
from .pointcloud import point_cloud, estimate_azimuth, POINT_DTYPE
from .microdoppler import slow_time, spectrogram, Spectrogram, doppler_extent

__all__ = [n for n in dir() if not n.startswith("_")]

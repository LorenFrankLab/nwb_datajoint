import datajoint as dj
import numpy as np
import nwb_datajoint as nd
from nwb_datajoint.common.common_nwbfile import AnalysisNwbfile
from position_tools import (get_angle, get_centriod, get_distance, get_speed,
                            get_velocity, interpolate_nan)
from pynwb.behavior import BehavioralTimeSeries, CompassDirection, Position

schema = dj.schema('common_position')


@schema
class PositionInfoParameters(dj.Lookup):
    definition = """
    param_name : varchar(80)        # name for this set of parameters
    ---
    max_separation = 9.0  : float   # max distance (in cm) between head LEDs
    max_speed = 300.0     : float   # max speed (in cm / s) of animal
    smoothing_std = 0.200 : float   # smoothing standard deviation (in seconds)
    """


@schema
class PositionInfo(dj.Computed):
    definition = """
    -> nd.common_session.Session
    -> PositionInfoParameters
    ---
    -> AnalysisNwbfile
    """
    # position info goes in nwb file, use nwb objects/object IDs

    def make(self, key):
        print(f'Populating for: {key}')
        key['analysis_file_name'] = AnalysisNwbfile().create(
            key['nwb_file_name'])

        raw_position = (nd.common_behav.RawPosition() &
                        {'nwb_file_name': key['nwb_file_name']}).fetch_nwb()
        raw_position = raw_position[0]['raw_position']

        position_info_parameters = (PositionInfoParameters() & key).fetch()

        head_position = Position()
        head_orientation = CompassDirection()
        head_velocity = BehavioralTimeSeries()
        head_speed = BehavioralTimeSeries()

        for ind, series_name in enumerate(raw_position.spatial_series):
            try:
                spatial_series = raw_position.get_spatial_series(series_name)
                position_info = self.calculate_position_info_from_spatial_series(
                    spatial_series,
                    position_info_parameters['max_separation'],
                    position_info_parameters['max_speed'],
                    position_info_parameters['smoothing_std'])
                head_position.create_spatial_series(
                    name=f"epoch_{ind:02d}",
                    timestamps=position_info['time'],
                    conversion=0.01,
                    data=position_info['head_position'],
                    reference_frame=spatial_series.reference_frame,
                    comments=spatial_series.comments,
                    description='head_x_position, head_y_position'
                )

                head_orientation.create_spatial_series(
                    name=f"epoch_{ind:02d}",
                    timestamps=position_info['time'],
                    conversion=1.0,
                    data=position_info['head_orientation'],
                    reference_frame=spatial_series.reference_frame,
                    comments=spatial_series.comments,
                    description='head_orientation'
                )

                head_velocity.create_timeseries(
                    name=f"epoch_{ind:02d}",
                    timestamps=position_info['time'],
                    conversion=0.01,
                    data=position_info['velocity'],
                    comments=spatial_series.comments,
                    description='head_x_velocity, head_y_velocity'
                )

                head_speed.create_timeseries(
                    name=f"epoch_{ind:02d}",
                    timestamps=position_info['time'],
                    conversion=0.01,
                    data=position_info['speed'],
                    comments=spatial_series.comments,
                    description='head_speed')
            except ValueError:
                pass

        nwb_analysis_file = AnalysisNwbfile()
        # processing module
        key['head_position_object_id'] = nwb_analysis_file.add_nwb_object(
            key['analysis_file_name'], head_position)
        key['head_orientation_object_id'] = nwb_analysis_file.add_nwb_object(
            key['analysis_file_name'], head_orientation)
        key['head_velocity_object_id'] = nwb_analysis_file.add_nwb_object(
            key['analysis_file_name'], head_velocity)
        key['head_speed_object_id'] = nwb_analysis_file.add_nwb_object(
            key['analysis_file_name'], head_speed)

        self.insert1(key)

    def calculate_position_info_from_spatial_series(
        self,
        spatial_series,
        max_LED_separation,
        max_plausible_speed,
        speed_smoothing_std_dev
    ):

        CM_TO_METERS = 100

        # Get spatial series properties
        time = spatial_series.timestamps  # seconds
        position = spatial_series.data  # meters
        dt = np.nanmean(np.diff(time))
        sampling_frequency = 1 / dt
        meters_to_pixels = spatial_series.conversion

        # Define LEDs
        back_LED = position[:, [0, 1]]
        front_LED = position[:, [2, 3]]

        # Convert to cm
        back_LED = back_LED * meters_to_pixels * CM_TO_METERS
        front_LED = front_LED * meters_to_pixels * CM_TO_METERS

        # Remove bad points
        interpolated_back_LED = back_LED.copy()
        interpolated_front_LED = front_LED.copy()

        # Calculate speed and remove bad points
        dist_between_LEDs = get_distance(back_LED, front_LED)
        is_too_separated = dist_between_LEDs >= max_LED_separation

        interpolated_back_LED[is_too_separated] = np.nan
        interpolated_front_LED[is_too_separated] = np.nan

        # Calculate speed and remove bad points
        front_LED_speed = get_speed(
            front_LED,
            time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_frequency)
        back_LED_speed = get_speed(
            back_LED,
            time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_frequency)

        is_too_fast = (front_LED_speed > max_plausible_speed) | (
            back_LED_speed > max_plausible_speed)
        interpolated_back_LED[is_too_fast] = np.nan
        interpolated_front_LED[is_too_fast] = np.nan

        # Interpolate
        interpolated_back_LED = interpolate_nan(interpolated_back_LED)
        interpolated_front_LED = interpolate_nan(interpolated_front_LED)

        # Smooth
        # smooth_back_LED = bottleneck.move_mean(
        #     interpolated_back_LED,
        #     window=moving_average_window,
        #     axis=0,
        #     min_count=1)
        # smooth_front_LED = bottleneck.move_mean(
        #     interpolated_front_LED,
        #     window=moving_average_window,
        #     axis=0, min_count=1)
        smooth_back_LED = interpolated_back_LED.copy()
        smooth_front_LED = interpolated_front_LED.copy()

        # Calculate head position, head orientation, velocity, speed
        head_position = get_centriod(smooth_back_LED, smooth_front_LED)  # cm
        head_orientation = get_angle(
            smooth_back_LED, smooth_front_LED)  # radians
        velocity = get_velocity(
            head_position,
            time=time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_frequency)  # cm/s
        speed = get_speed(
            head_position,
            time=time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_frequency)  # cm/s

        return {'time': time,
                'head_position': head_position,
                'head_orientation': head_orientation,
                'velocity': velocity,
                'speed': speed,
                }


# @schema
# class LinearizationParameters(dj.Lookup):
#     definition = """
#     param_name : varchar(80)   # name for this set of parameters
#     ---
#     use_HMM = False  : float   # use HMM to determine linearization
#     route_euclidean_distance_scaling = 1.0 : float   # How much to prefer route distances between successive time points that are closer to the euclidean distance. Smaller numbers mean the route distance is more likely to be close to the euclidean distance.
#     sensor_std_dev = 5.0 : float   # Uncertainty of position sensor (in cm).
#     diagonal_bias = 0.5 : float   # Biases the transition matrix to prefer the current track segment.
#     """
#
#
# @schema
# class LinearizedPosition(dj.Computed):
#     definition = """
#     # Table for holding spike sorting runs
#     -> LinearizationParameters
#     ---
#     -> AnalysisNwbfile
#     units_object_id: varchar(40)           # Object ID for the units in NWB file
#     time_of_sort=0: int                    # This is when the sort was done
#     curation_feed_uri='': varchar(1000)    # Labbox-ephys feed for curation
    # """

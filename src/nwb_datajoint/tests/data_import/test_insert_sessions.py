import datetime
from hdmf.backends.warnings import BrokenLinkWarning
import pathlib
import pynwb
import pytest
import shutil
import os

from nwb_datajoint.data_import.insert_sessions import copy_nwb_link_raw_ephys


@pytest.fixture()
def new_nwbfile_raw_file_name():
    nwbfile = pynwb.NWBFile(
        session_description='session_description',
        identifier='identifier',
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    ts = pynwb.TimeSeries(
        name='test_ts',
        data=[1, 2, 3],
        unit='unit',
        timestamps=[1., 2., 3.],
    )
    nwbfile.add_acquisition(ts)

    base_dir = pathlib.Path(os.getenv('NWB_DATAJOINT_BASE_DIR', None))
    assert base_dir is not None, 'You must set NWB_DATAJOINT_BASE_DIR or provide the base_dir argument'

    file_name = 'raw.nwb'
    file_path = base_dir / file_name
    with pynwb.NWBHDF5IO(str(file_path), mode='w') as io:
        io.write(nwbfile)

    return file_name


@pytest.fixture()
def new_nwbfile_no_ephys_file_name():
    return 'raw_no_ephys.nwb'


@pytest.fixture()
def moved_nwbfile_no_ephys_file_path(tmp_dir, new_nwbfile_no_ephys_file_name):
    return tmp_dir / new_nwbfile_no_ephys_file_name


def test_copy_nwb(new_nwbfile_raw_file_name, new_nwbfile_no_ephys_file_name, moved_nwbfile_no_ephys_file_path):
    copy_nwb_link_raw_ephys(new_nwbfile_raw_file_name, new_nwbfile_no_ephys_file_name)

    base_dir = pathlib.Path(os.getenv('NWB_DATAJOINT_BASE_DIR', None))
    out_nwb_file_abspath = base_dir / new_nwbfile_no_ephys_file_name
    with pynwb.NWBHDF5IO(path=str(out_nwb_file_abspath), mode='r') as io:
        nwbfile = io.read()
        assert 'test_ts' not in nwbfile.acquisition

    # test readability after move
    shutil.move(out_nwb_file_abspath, moved_nwbfile_no_ephys_file_path)
    with pynwb.NWBHDF5IO(path=str(moved_nwbfile_no_ephys_file_path), mode='r') as io:
        with pytest.warns(BrokenLinkWarning):
            nwbfile = io.read()  # should raise BrokenLinkWarning
        assert 'test_ts' not in nwbfile.acquisition

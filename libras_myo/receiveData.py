from __future__ import print_function
import myo as libmyo
import time
import sys
import csv
import argparse
import os
import errno

nameOfFileEmg = None
nameOfFileOrientation = None
nameOfFileAccelerometor = None

class Listener(libmyo.DeviceListener):
    """
    Listener implementation. Return False from any function to
    stop the Hub.
    """

    interval = 0.05  # Output only 0.05 seconds

    def clearFiles(self):
        f = open(nameOfFileEmg, 'w+').close()
        f = open(nameOfFileOrientation, 'w+').close()
        f = open(nameOfFileAccelerometor, 'w+').close()

    def __init__(self):
        super(Listener, self).__init__()
        self.orientation = None
        self.acceleration = None
        self.pose = libmyo.Pose.rest
        self.emg_enabled = False
        self.locked = False
        self.rssi = None
        self.emg = None
        self.timestamp = None
        self.last_time = 0
        # self.clearFiles()

    def on_connect(self, myo, timestamp, firmware_version):
        myo.vibrate('short')
        myo.vibrate('short')
        myo.request_rssi()
        myo.request_battery_level()
        myo.set_stream_emg(libmyo.StreamEmg.enabled)
        self.emg_enabled = True

    def on_rssi(self, myo, timestamp, rssi):
        pass

    def on_orientation_data(self, myo, timestamp, orientation):
        self.orientation = orientation
        self.timestamp = timestamp
        self.outputData(2)

    def on_accelerometor_data(self, myo, timestamp, acceleration):
        self.acceleration = acceleration
        self.timestamp = timestamp
        self.outputData(1)

    def on_gyroscope_data(self, myo, timestamp, gyroscope):
        pass

    def on_emg_data(self, myo, timestamp, emg):
        self.emg = emg
        self.timestamp = timestamp
        self.outputData(3)

    def on_unlock(self, myo, timestamp):
        pass

    def on_lock(self, myo, timestamp):
        pass

    def on_event(self, kind, event):
        """
        Called before any of the event callbacks.
        """

    def on_event_finished(self, kind, event):
        """
        Called after the respective event callbacks have been
        invoked. This method is *always* triggered, even if one of
        the callbacks requested the stop of the Hub.
        """

    def on_pair(self, myo, timestamp, firmware_version):
        """
        Called when a Myo armband is paired.
        """

    def on_unpair(self, myo, timestamp):
        """
        Called when a Myo armband is unpaired.
        """

    def on_disconnect(self, myo, timestamp):
        """
        Called when a Myo is disconnected.
        """

    def on_arm_sync(self, myo, timestamp, arm, x_direction, rotation,
                    warmup_state):
        """
        Called when a Myo armband and an arm is synced.
        """

    def on_arm_unsync(self, myo, timestamp):
        """
        Called when a Myo armband and an arm is unsynced.
        """

    def on_battery_level_received(self, myo, timestamp, level):
        """
        Called when the requested battery level received.
        """

    def on_warmup_completed(self, myo, timestamp, warmup_result):
        """
        Called when the warmup completed.
        """

    ## ----------------------------------------
    ## ---------- HELPER METHODS --------------
    ## ----------------------------------------

    def writeData(self, nameOfFile, data):
        with open(nameOfFile, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)


    def createArray(self, type):

        parts = []

        if self.timestamp:
                parts.append(str(self.timestamp))

        if type == 1:
            if self.acceleration:
                for comp in self.acceleration:
                    parts.append(str(comp))

        elif type == 2:
            if self.orientation:
                for comp in self.orientation:
                    parts.append(str(comp))

        else:
            if self.emg:
                for comp in self.emg:
                    parts.append(str(comp))

        return parts


    def outputData(self, type):
            dataArray = self.createArray(type)

            filePath = None
            if type == 1:
                filePath = nameOfFileAccelerometor
            elif type == 2:
                filePath = nameOfFileOrientation
            else:
                filePath = nameOfFileEmg

            self.writeData(filePath, dataArray)


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def main(args):

    make_sure_path_exists("data/%s" % args.user)

    global nameOfFileEmg
    nameOfFileEmg = "data/%s/%s-%s-%s-emg.csv" % (args.user, args.user,  (args.letter).capitalize(), args.turn)
    global nameOfFileOrientation
    nameOfFileOrientation = "data/%s/%s-%s-%s-orientation.csv" % (args.user, args.user, (args.letter).capitalize(), args.turn)
    global nameOfFileAccelerometor
    nameOfFileAccelerometor = "data/%s/%s-%s-%s-acc.csv" % (args.user, args.user, (args.letter).capitalize(), args.turn)

    libmyo.init()
    print("Connecting to Myo ... Use CTRL^C to exit.")
    try:
        hub = libmyo.Hub()
    except MemoryError:
        print("Myo Hub could not be created. \
              Make sure Myo Connect is running.")
        return

    hub.set_locking_policy(libmyo.LockingPolicy.none)
    listener = Listener()

    # Listen to keyboard interrupts and stop the hub in that case.
    try:
        while True:
            hub.run_once(1000/200, listener)
    except KeyboardInterrupt:
        print("\nQuitting ...")
    finally:
        print("Shutting down hub...")
        hub.shutdown()


def runArgs(args):
    nameOfFileEmg = "%s-%s-%s-emg.csv" % (args.user, (args.letter).capitalize(), args.turn)
    nameOfFileOrientation = "%s-%s-%s-orientation.csv" % (args.user, (args.letter).capitalize(), args.turn)
    nameOfFileAccelerometor = "%s-%s-%s-acc.csv" % (args.user, (args.letter).capitalize(), args.turn)

if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument("user", help="The name of the user.")
    parser.add_argument("letter", help="The letter the user is doing.")
    parser.add_argument("turn", help="The number of the turn.")
    args = parser.parse_args()
    main(args)

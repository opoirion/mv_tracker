 # -*- coding: utf-8 -*-

"""main function to launch a webcam tracker movement """

from mv_tracker.cam_launcher import CamLauncher


def main():
    cam_launcher = CamLauncher()
    cam_launcher.run()


if __name__ == "__main__":
        main()

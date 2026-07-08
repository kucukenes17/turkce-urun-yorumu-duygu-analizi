"""pytest kök yapılandırması.

Bu dosyanın varlığı, pytest'in repo kökünü `sys.path`'e eklemesini sağlar;
böylece testlerdeki `from src import ...` içe aktarmaları CI ortamında da çalışır.
"""

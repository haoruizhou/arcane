# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

# Add models directory to datas
# Format is (source_path, destination_dir_inside_bundle)
datas = [
    ('models', 'models'),
]

# If you have assets later, add them here:
# datas += [('assets', 'assets')]

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'ultralytics', 
        'torchvision', 
        'rawpy',
        'huggingface_hub'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorboard',
        'torch.utils.tensorboard',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='arcane',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Set to True for debugging console output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='arcane',
)
app = BUNDLE(
    coll,
    name='arcane.app',
    icon=None, # Add an icon path here if you have one
    bundle_identifier=None,
)

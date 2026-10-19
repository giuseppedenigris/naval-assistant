"""Copy the autopilot sample into raw/ with every section labelled [§n], n being the position of its header in the document."""

import re
from itertools import count
from pathlib import Path

KB_DIR = Path(r"C:\Users\peppe\Desktop\coding\pdf-ingestion\data\output")
RAW_DIR = Path(__file__).parent / "raw"
HEADER = re.compile(r"^(#{1,6})[ \t]+", re.M)

SAMPLE = [
    "GHP_10_Install_EN.md",
    "GHP10MarineAutopilotSystem_InstallationInstructions.md",
    "Reactor_40_Mechanical_Configuration_Guide_EN.md",
    "Simrad_AP24_Manual_EN_web.md",
    "simrad-ap70-ap80-autopilot-control-head-user-manual.md",
    "1001-Installation-Operation-Manual-v0r0.md",
    "48565OM.md",
    "eng-comnav-1440-1460-manu.md",
    "Evolution_EV-2_Installation_instructions.md",
    "17223617121621860762SP70___80_Manual.md",
    "3566_NP2015_25_OP.md",
    "KAP866_OME_130206.md",
    "ha1205-installation-manual.md",
    "Operation-Manual_4JH3DTE-49961-202851.md",
    "Sea_Recovery_Marine_Brochure.md",
    "Ace-Marine-Diesel-MAN-Technical-Data-Sheets.md",
    "Latis-Ops-Manual.md",
]


def label_sections(md: str) -> str:
    n = count(1)
    return HEADER.sub(lambda m: f"{m.group(1)} [§{next(n)}] ", md)


def main() -> None:
    RAW_DIR.mkdir(exist_ok=True)
    for name in SAMPLE:
        (RAW_DIR / name).write_text(label_sections((KB_DIR / name).read_text(encoding="utf-8")), encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()

all: SRAD_V1 SRAD_V2

SRAD_V1:
	cd srad_v1 && nmake /f Makefile

SRAD_V2:
	cd srad_v2 && nmake /f Makefile

clean: SRAD_V1_clean SRAD_V2_clean

SRAD_V1_clean:
	cd srad_v1 && nmake /f Makefile clean

SRAD_V2_clean:
	cd srad_v2 && nmake /f Makefile clean
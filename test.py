def test_missing_incidence_angle_raises(SAR_metadata_extractor: SARMetadataExtractor) -> None:
    class SCPOCA_Missing(Exception):
        pass

    class SICD_MissingIncidence:
        def __init__(self):
            self.CollectionInfo = {"CoreName": "X"}
            self.ImageCreation = {"DateTime": "2020-01-01T00:00:00"}
            self.RadarMode = {"ModeType": "Spotlight"}
            self.GeoData = {"SCP": {"ECF": [0, 0, 0]}}
            self.SCP0CA = None  # missing field triggers ValueError

    # monkeypatch the parser to return mock object
    SAR_metadata_extractor.SICDType = type("MockSICDType", (), {"from_xml_file": staticmethod(lambda _: SICD_MissingIncidence())})

    fake_path = Path("/fake/path/test_file.xml")

    with pytest.raises(ValueError, match="Missing SCPOCA.IncidenceAng"):
        SAR_metadata_extractor.extract_metadata(fake_path)

def test_missing_llhpoint_raises(SAR_metadata_extractor: SARMetadataExtractor, tmp_path: Path) -> None:
    class SICD_MissingLLH:
        def __init__(self):
            self.CollectionInfo = {"CoreName": "Test"}
            self.ImageCreation = {"DateTime": "2020-01-01T00:00:00"}
            self.RadarMode = {"ModeType": "Spotlight"}
            self.SCPOA = {"IncidenceAng": 45.0}
            self.GeoData = {}  # missing SCP

    file_path = tmp_path / "test_SICD.xml"
    file_path.write_text("<dummy></dummy>")

    SAR_metadata_extractor.directory_path = tmp_path
    SAR_metadata_extractor.SICDType = type("MockSICDType", (), {"from_xml_file": staticmethod(lambda _: SICD_MissingLLH())})

    metadata = SAR_metadata_extractor.parse_sicd_metadata()

    assert file_path.name not in metadata

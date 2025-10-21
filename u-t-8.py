# ---------------------------- TEST IMPLEMENTATIONS ----------------------------

class DummyDataAdapter:
    """The DataAdapter for the Dummy Model"""

    @staticmethod
    def to_model(input: Any, target: dict) -> np.ndarray:
        """Converts from sdk inputs to dummy model inputs"""
        return np.asarray([x[ATRSampleKeys.CHIP_INDEX] for x in target])

    @staticmethod
    def from_model(model_outputs: tuple) -> tuple:
        """Converts from dummy model outputs to taika outputs"""
        bboxes, chip_ids, labels, scores = model_outputs
        return chip_ids, bboxes, labels, scores


class DummyModel(ModelInterface):
    """A dummy model that includes a `batch_infer` method."""

    def __init__(self):
        """Initialization"""
        self.bboxes = np.asarray([[0, 0, 10, 10], [20, 20, 15, 15]])
        self.labels = np.asarray([1, 2])
        self._data_adapter = DummyDataAdapter()

    def batch_infer(self, chip_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Creates dummy detections for a batch of chip_ids"""
        bboxes = [self.bboxes + x for x in chip_ids]
        labels = [self.labels + x for x in chip_ids]
        chip_ids = np.repeat(chip_ids, len(self.bboxes))
        bboxes = np.concatenate(bboxes)
        labels = np.concatenate(labels)
        scores = np.asarray([1.0] * len(bboxes))
        return bboxes, chip_ids, labels, scores

    @property
    def data_adapter(self):
        """Returns the model's data adapter"""
        return self._data_adapter


# ---------------------------- FIXTURES ----------------------------

@pytest.fixture
def dummy_model() -> ModelInterface:
    """Creates a dummy model instance"""
    return DummyModel()


@pytest.fixture
def class_map() -> dict[int, str]:
    """Creates a class map for testing"""
    return {1: "class1", 2: "class2", 3: "class3"}


@pytest.fixture
def mock_preprocessor_wizard(preprocessing_config: KVHolder) -> ImagePreprocessorAndChipperWizard:
    """Creates a mock preprocessor wizard using the preprocessing config from conftest"""
    class TestWizard(ImagePreprocessorAndChipperWizard):
        def __init__(self, config):
            self.config = config

        def create_preprocessor(self, image: np.ndarray, image_metadata) -> ImagePreprocessorAndChipper:
            """Create a test preprocessor"""
            class TestPreprocessor(ImagePreprocessorAndChipper):
                def __init__(self):
                    self._resampler = None
                    self._image_sampler = None

                def __len__(self):
                    return 9

                def __getitem__(self, idx):
                    return np.zeros((100, 100)), {
                        ATRSampleKeys.CHIP_INDEX: idx,
                        ATRSampleKeys.BOUNDARY: [0, 0, 100, 100]
                    }

            return TestPreprocessor()

    return TestWizard(preprocessing_config)


@pytest.fixture
def atr_inferencer(
    dummy_model: ModelInterface,
    mock_preprocessor_wizard: ImagePreprocessorAndChipperWizard,
    class_map: dict[int, str]
) -> ATRInferencer:
    """Creates an ATR Inferencer instance"""
    augmentations = [MinMaxNorm(return_0_on_static_input=True)]
    return ATRInferencer(
        model=dummy_model,
        preprocessor_wizard=mock_preprocessor_wizard,
        augmentations=augmentations,
        class_map=class_map,
        batch_size=2,
        num_workers=0,
        score_threshold=0.5,
        perform_nms=True,
        nms_threshold=0.5
    )


@pytest.fixture
def sample_image_path(taika_tests_path: Path) -> Path:
    """Creates a mock image file path for testing using the test path from conftest"""
    image_path = taika_tests_path / "test_image.ntif"
    image_path.touch()  # Create the file so it exists
    return image_path


@pytest.fixture
def full_inference_config(preprocessing_config: KVHolder) -> KVHolder:
    """Creates a complete inference configuration using preprocessing config from conftest"""
    return KVHolder({
        "data_preprocessing": preprocessing_config.to_dict(),
        "class_map": {1: "class1", 2: "class2", 3: "class3"},
        "inference": {
            "environment": {
                "test_env": {
                    "batch_size": 4,
                    "num_workers": 2
                }
            },
            "dataloader": {
                "transforms": {}
            },
            "nms": {
                "enabled": True,
                "nms_threshold": 0.5
            },
            "score_threshold": 0.7
        }
    })


# ---------------------------- TESTS ----------------------------

def test_atr_inferencer_init(atr_inferencer: ATRInferencer) -> None:
    """Tests ATRInferencer initialization"""
    assert atr_inferencer is not None
    assert atr_inferencer._model is not None
    assert atr_inferencer._preprocessor_wizard is not None
    assert atr_inferencer._class_map is not None
    assert atr_inferencer._batch_size == 2
    assert atr_inferencer._num_workers == 0
    assert atr_inferencer._score_threshold == 0.5
    assert atr_inferencer._perform_nms is True
    assert atr_inferencer._nms_threshold == 0.5


def test_atr_inferencer_infer_file_not_found(atr_inferencer: ATRInferencer) -> None:
    """Tests that infer raises FileNotFoundError for non-existent files"""
    non_existent_path = Path("/non/existent/file.ntif")
    with pytest.raises(FileNotFoundError):
        atr_inferencer.infer(non_existent_path)


def test_atr_inferencer_invalid_image_path(atr_inferencer: ATRInferencer) -> None:
    """Tests that TypeError is raised for invalid image path types"""
    with pytest.raises(TypeError):
        atr_inferencer.infer(123)
    with pytest.raises(TypeError):
        atr_inferencer.infer(["/some/path"])


def test_atr_inferencer_algorithm_metadata(atr_inferencer: ATRInferencer) -> None:
    """Tests algorithm metadata property"""
    assert atr_inferencer.algorithm_metadata == {}
    new_metadata = {"version": "1.0", "name": "test"}
    atr_inferencer.algorithm_metadata = new_metadata
    assert atr_inferencer.algorithm_metadata == new_metadata


def test_atr_inferencer_boxes_to_polygon(atr_inferencer: ATRInferencer, sicd_metadata: SICDType) -> None:
    """Tests the _boxes_to_polygon method using SICD metadata from conftest"""
    # Test with boxes in correct format
    bboxes = np.array([
        [10, 20, 30, 40],  # x, y, w, h
        [50, 60, 70, 80]
    ])
    
    polygons = atr_inferencer._boxes_to_polygon(bboxes, sicd_metadata)
    
    assert isinstance(polygons, np.ndarray)
    assert polygons.shape == (2, 4, 2)  # 2 boxes, 4 corners each, 2 coordinates
    
    num_cols = sicd_metadata.ImageData.NumCols
    num_rows = sicd_metadata.ImageData.NumRows
    
    # Verify clipping works
    assert np.all(polygons[:, :, 0] >= 0)
    assert np.all(polygons[:, :, 0] < num_cols)
    assert np.all(polygons[:, :, 1] >= 0)
    assert np.all(polygons[:, :, 1] < num_rows)
    
    # Test with empty boxes
    empty_boxes = np.array([])
    empty_polygons = atr_inferencer._boxes_to_polygon(empty_boxes, sicd_metadata)
    assert empty_polygons.size == 0


def test_atr_inferencer_boxes_to_polygon_with_sidd(atr_inferencer: ATRInferencer, sidd_metadata: SIDDType) -> None:
    """Tests the _boxes_to_polygon method using SIDD metadata from conftest"""
    # Test with boxes that might exceed boundaries
    bboxes = np.array([
        [10000, 10000, 5000, 5000],  # Potentially exceeds bounds
        [50, 50, 100, 100]  # Within bounds
    ])
    
    polygons = atr_inferencer._boxes_to_polygon(bboxes, sidd_metadata[0])  # sidd_metadata is a list
    
    assert isinstance(polygons, np.ndarray)
    assert polygons.shape == (2, 4, 2)
    
    # Get dimensions from SIDD metadata
    num_cols = sidd_metadata[0].Measurement.PixelFootprint.Col
    num_rows = sidd_metadata[0].Measurement.PixelFootprint.Row
    
    # Verify all values are properly clipped
    assert np.all(polygons[:, :, 0] >= 0)
    assert np.all(polygons[:, :, 0] < num_cols)
    assert np.all(polygons[:, :, 1] >= 0)
    assert np.all(polygons[:, :, 1] < num_rows)


def test_atr_inferencer_file_exists(atr_inferencer: ATRInferencer, sample_image_path: Path) -> None:
    """Tests the _file_exists method"""
    # Test with existing file (created by fixture)
    atr_inferencer._file_exists(sample_image_path)  # Should not raise
    
    # Test with non-existing file
    with pytest.raises(FileNotFoundError):
        atr_inferencer._file_exists(Path("/non/existent/file.ntif"))


def test_atr_inferencer_call_method(
    atr_inferencer: ATRInferencer,
    sample_image_path: Path,
    mock_sicd_reader: SICDReader,
    mocker
) -> None:
    """Tests that __call__ method works as alias for infer"""
    mock_image = np.zeros(mock_sicd_reader.data_segment.formatted_shape, 
                          dtype=mock_sicd_reader.data_segment.formatted_dtype)
    
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.read_sicd', 
                 return_value=(mock_image, mock_sicd_reader.sicd_meta))
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.SingleImageDataset')
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.atr_image_inference',
                 return_value=(np.array([[10, 10, 20, 20]]), np.array([1]), np.array([0.9])))
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.pixel_to_llh',
                 side_effect=lambda x, y: x)
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.atr_to_geojson',
                 return_value={"type": "FeatureCollection", "features": []})
    
    result = atr_inferencer(sample_image_path)
    
    assert isinstance(result, dict)
    assert result["type"] == "FeatureCollection"


def test_atr_inferencer_load_from_config(
    dummy_model: ModelInterface, 
    full_inference_config: KVHolder, 
    mocker
) -> None:
    """Tests loading ATRInferencer from config using the full config fixture"""
    mock_wizard = mocker.MagicMock(spec=ImagePreprocessorAndChipperWizard)
    mocker.patch.object(ImagePreprocessorAndChipperWizard, 'from_preprocessor_config', 
                       return_value=mock_wizard)
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.build_transforms_from_config', 
                 return_value=[])
    
    inferencer = ATRInferencer.load_from_config(
        model=dummy_model, 
        config=full_inference_config, 
        environment_name="test_env"
    )
    
    assert isinstance(inferencer, ATRInferencer)
    assert inferencer._batch_size == 4
    assert inferencer._num_workers == 2
    assert inferencer._score_threshold == 0.7
    assert inferencer._perform_nms is True
    assert inferencer._nms_threshold == 0.5


def test_inference_config_kvholder(preprocessing_config: KVHolder) -> None:
    """Tests that inference config works as KVHolder using preprocessing_config from conftest"""
    config = KVHolder({
        "batch_size": 4,
        "num_workers": 2,
        "pin_memory": False,
        "score_threshold": 0.7,
        "perform_nms": True,
        "nms_threshold": 0.3
    })
    
    assert config.batch_size == 4
    assert config.num_workers == 2
    assert config.pin_memory is False
    assert config.score_threshold == 0.7
    assert config.perform_nms is True
    assert config.nms_threshold == 0.3

    config_dict = KVHolder({"score_threshold": {1: 0.5, 2: 0.6, 3: 0.7}})
    assert isinstance(config_dict.score_threshold, dict)
    assert config_dict.score_threshold[1] == 0.5
    assert config_dict.score_threshold[2] == 0.6
    assert config_dict.score_threshold[3] == 0.7

    assert preprocessing_config.chipping is not None
    assert preprocessing_config.processing_steps is not None
    assert preprocessing_config.dataset_creation is not None

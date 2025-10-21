# ---------------------------- FIXTURES ----------------------------

@pytest.fixture
def dummy_model() -> ModelInterface:
    """A dummy model that includes a `batch_infer` method."""
    
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
    
    return DummyModel()


@pytest.fixture
def class_map() -> dict[int, str]:
    """Creates a class map for testing"""
    return {1: "class1", 2: "class2", 3: "class3"}


@pytest.fixture
def mock_preprocessor_wizard(preprocessing_config: KVHolder):
    """Creates a mock preprocessor wizard"""
    wizard = MagicMock(spec=Noita)
    
    # Mock the create_preprocessor method to return a mock preprocessor
    mock_preprocessor = MagicMock()
    mock_preprocessor._resampler = None
    wizard.create_preprocessor.return_value = mock_preprocessor
    
    return wizard


@pytest.fixture
def atr_inferencer(
    dummy_model: ModelInterface,
    mock_preprocessor_wizard: Noita,
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
    """Creates a mock image file path for testing"""
    # Create a dummy file for testing
    image_path = taika_tests_path / "test_image.ntif"
    image_path.touch()
    return image_path


# ---------------------- TESTS ----------------------

def test_atr_inferencer_init(atr_inferencer: ATRInferencer) -> None:
    """Tests ATRInferencer initialization"""
    assert atr_inferencer is not None
    assert atr_inferencer._model is not None
    assert atr_inferencer._preprocessor_wizard is not None
    assert atr_inferencer._class_map is not None
    assert atr_inferencer._batch_size == 2
    assert atr_inferencer._num_workers == 0
    assert atr_inferencer._score_threshold == 0.5
    assert atr_inferencer._perform_nms == True
    assert atr_inferencer._nms_threshold == 0.5
    return


def test_atr_inferencer_infer_with_mock(
    atr_inferencer: ATRInferencer,
    sample_image_path: Path,
    mocker
) -> None:
    """Tests performing inference with mocked components"""
    # Mock the read_sicd function
    mock_image = np.zeros((100, 100), dtype=np.complex64)
    mock_metadata = MagicMock()
    mock_metadata.ImageData.NumRows = 100
    mock_metadata.ImageData.NumCols = 100
    
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.read_sicd', 
                 return_value=(mock_image, mock_metadata))
    
    # Mock SingleImageDataset
    mock_dataset = MagicMock()
    mock_dataset.__len__ = lambda: 9  # 9 chips
    mock_dataset.image_preprocessor = MagicMock()
    mock_dataset.image_preprocessor._resampler = None
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.SingleImageDataset', 
                 return_value=mock_dataset)
    
    # Mock atr_image_inference
    mock_boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
    mock_labels = np.array([1, 2])
    mock_scores = np.array([0.9, 0.8])
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.atr_image_inference',
                 return_value=(mock_boxes, mock_labels, mock_scores))
    
    # Mock pixel_to_llh
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.pixel_to_llh',
                 side_effect=lambda x, y: x)  # Just return the input
    
    # Mock atr_to_geojson
    expected_geojson = {"type": "FeatureCollection", "features": []}
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.atr_to_geojson',
                 return_value=expected_geojson)
    
    # Run inference
    result = atr_inferencer.infer(sample_image_path)
    
    assert result == expected_geojson
    assert isinstance(result, dict)
    return


def test_atr_inferencer_algorithm_metadata(atr_inferencer: ATRInferencer) -> None:
    """Tests algorithm metadata property"""
    # Test getter
    assert atr_inferencer.algorithm_metadata == {}
    
    # Test setter
    new_metadata = {"version": "1.0", "name": "test"}
    atr_inferencer.algorithm_metadata = new_metadata
    assert atr_inferencer.algorithm_metadata == new_metadata
    return


def test_atr_inferencer_boxes_to_polygon(
    atr_inferencer: ATRInferencer
) -> None:
    """Tests the _boxes_to_polygon method"""
    # Create mock metadata
    mock_metadata = MagicMock()
    mock_metadata.ImageData.NumRows = 1000
    mock_metadata.ImageData.NumCols = 1200
    
    # Test with normal boxes
    bboxes = np.array([
        [10, 20, 30, 40],  # x, y, w, h
        [100, 150, 50, 60]
    ])
    
    polygons = atr_inferencer._boxes_to_polygon(bboxes, mock_metadata)
    
    assert isinstance(polygons, np.ndarray)
    assert polygons.shape == (2, 4, 2)  # 2 boxes, 4 corners each, 2 coordinates
    
    # Test with empty boxes
    empty_boxes = np.array([])
    empty_polygons = atr_inferencer._boxes_to_polygon(empty_boxes, mock_metadata)
    assert empty_polygons.shape == (0,)
    return


def test_atr_inferencer_file_exists(
    atr_inferencer: ATRInferencer,
    sample_image_path: Path
) -> None:
    """Tests the _file_exists method"""
    # Test with existing file
    atr_inferencer._file_exists(sample_image_path)  # Should not raise
    
    # Test with non-existing file
    with pytest.raises(FileNotFoundError):
        atr_inferencer._file_exists(Path("/non/existent/file.ntif"))
    return


def test_atr_inferencer_load_from_config(
    dummy_model: ModelInterface,
    mocker
) -> None:
    """Tests loading ATRInferencer from config"""
    # Create a mock config
    config = KVHolder({
        "data_preprocessing": {
            "chipping": {"params": {"chip_size": {"row": 512, "col": 512}}},
            "processing_steps": {"remap": {}, "resample": {}, "transformations": {}},
            "dataset_creation": {}
        },
        "class_map": {1: "class1", 2: "class2"},
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
    
    # Mock the Noita.from_preprocessor_config method
    mock_wizard = MagicMock(spec=Noita)
    mocker.patch.object(Noita, 'from_preprocessor_config', return_value=mock_wizard)
    
    # Mock build_transforms_from_config
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.build_transforms_from_config',
                 return_value=[])
    
    # Load from config
    inferencer = ATRInferencer.load_from_config(
        model=dummy_model,
        config=config,
        environment_name="test_env"
    )
    
    assert isinstance(inferencer, ATRInferencer)
    assert inferencer._batch_size == 4
    assert inferencer._num_workers == 2
    assert inferencer._score_threshold == 0.7
    assert inferencer._perform_nms == True
    assert inferencer._nms_threshold == 0.5
    return


def test_atr_inferencer_call_method(
    atr_inferencer: ATRInferencer,
    sample_image_path: Path,
    mocker
) -> None:
    """Tests that __call__ method works as alias for infer"""
    # Mock the infer method
    expected_result = {"type": "FeatureCollection"}
    mocker.patch.object(atr_inferencer, 'infer', return_value=expected_result)
    
    # Call using __call__
    result = atr_inferencer(sample_image_path)
    
    assert result == expected_result
    atr_inferencer.infer.assert_called_once_with(sample_image_path)
    return


def test_inference_config_kvholder() -> None:
    """Tests that inference config works as KVHolder"""
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
    assert config.pin_memory == False
    assert config.score_threshold == 0.7
    assert config.perform_nms == True
    assert config.nms_threshold == 0.3
    
    # Test with class-specific thresholds
    config_dict = KVHolder({
        "score_threshold": {1: 0.5, 2: 0.6, 3: 0.7}
    })
    
    assert isinstance(config_dict.score_threshold, dict)
    assert config_dict.score_threshold[1] == 0.5
    assert config_dict.score_threshold[2] == 0.6
    assert config_dict.score_threshold[3] == 0.7
    return

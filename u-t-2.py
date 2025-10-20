# ---------------------- FIXTURES ----------------------

@pytest.fixture
def inference_config() -> InferenceConfig:
    """Creates an inference configuration"""
    return InferenceConfig(
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        score_threshold=0.5,
        perform_nms=True,
        nms_threshold=0.5,
        augmentations=[MinMaxNorm(return_0_on_static_input=True)],
    )


@pytest.fixture
def dummy_model() -> ModelInterface:
    """A dummy model that includes a `batch_infer` method."""
    
    class DummyDataAdapter(DataAdapter):
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
            """Creates dummy detections for a batch of chip_ids
            
            Args:
                chip_ids (list[int]): A list of chip ids
                
            Returns:
                A tuple containing
                    - bboxes (ndarray): An array of bboxes in x,y,w,h order
                    - chip_ids (ndarray): The chip_id for which each detection belongs to
                    - labels (ndarray): The numerical label for each detection
                    - scores (ndarray): The score for each detection, always 1.0.
            """
            bboxes = [self.bboxes + x for x in chip_ids]
            labels = [self.labels + x for x in chip_ids]
            chip_ids = np.repeat(chip_ids, len(self.bboxes))
            bboxes = np.concatenate(bboxes)
            labels = np.concatenate(labels)
            scores = np.asarray([1.0] * len(bboxes))
            return bboxes, chip_ids, labels, scores
        
        @property
        def data_adapter(self) -> DataAdapter:
            """Returns the model's data adapter"""
            return self._data_adapter
    
    return DummyModel()


@pytest.fixture
def atr_inferencer(
    inference_config: InferenceConfig,
    dummy_model: ModelInterface,
    preprocessing_config: KVHolder
) -> ATRInferencer:
    """Creates an ATR Inferencer instance"""
    return ATRInferencer(
        model=dummy_model,
        preprocessing_config=preprocessing_config,
        inference_config=inference_config
    )


@pytest.fixture
def inference_preprocessor(
    preprocessing_config: KVHolder,
    sicd_metadata: SICDType,
    mocker: MockerFixture
) -> ImagePreprocessorAndChipper:
    """Creates a mock ImagePreprocessorAndChipper for inference"""
    preprocessor = mocker.MagicMock(ImagePreprocessorAndChipper)
    preprocessor.chip_size = (1024, 900)
    preprocessor.chip_stride = (900, 800)
    preprocessor._resampler = None
    preprocessor.metadata = sicd_metadata
    
    # Mock the chip generation
    preprocessor.get_chip_boundaries.return_value = [
        [100, 200, 1100, 1200],
        [1100, 1200, 2100, 2200],
        [500, 0, 1500, 900],
        [1000, 0, 2000, 900],
        [500, 500, 1500, 1400],
        [500, 500, 1500, 1400],
        [1000, 500, 2000, 1400],
        [1000, 1000, 2000, 1900],
        [500, 1000, 1500, 1900],
        [1000, 1000, 2000, 1900]
    ]
    
    return preprocessor


# ---------------------- TESTS ----------------------

def test_atr_inferencer_init(atr_inferencer: ATRInferencer) -> None:
    """Tests ATRInferencer initialization"""
    assert atr_inferencer is not None
    assert atr_inferencer.model is not None
    assert atr_inferencer.preprocessing_config is not None
    assert atr_inferencer.inference_config is not None
    assert isinstance(atr_inferencer.inference_config, InferenceConfig)


def test_atr_inferencer_from_sicd(
    atr_inferencer: ATRInferencer,
    mock_sicd_reader: SICDReader,
    mocker: MockerFixture
) -> None:
    """Tests performing inference from a SICD reader"""
    # Mock the internal inference method
    mock_inference = mocker.patch.object(
        atr_inferencer,
        '_run_inference',
        return_value=InferenceResult(
            bboxes=np.asarray([[100, 110, 25, 30], [20, 20, 15, 15]]),
            labels=np.asarray([1, 2]),
            scores=np.asarray([1.0, 1.0])
        )
    )
    
    result = atr_inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(result, InferenceResult)
    assert isinstance(result.bboxes, np.ndarray)
    assert isinstance(result.labels, np.ndarray)
    assert isinstance(result.scores, np.ndarray)
    assert len(result.bboxes) == 2
    assert len(result.labels) == 2
    assert len(result.scores) == 2
    mock_inference.assert_called_once()


def test_atr_inferencer_from_sidd(
    atr_inferencer: ATRInferencer,
    mock_sidd_reader: SIDDReader,
    mocker: MockerFixture
) -> None:
    """Tests performing inference from a SIDD reader"""
    # Mock the internal inference method
    mock_inference = mocker.patch.object(
        atr_inferencer,
        '_run_inference',
        return_value=InferenceResult(
            bboxes=np.asarray([[100, 110, 25, 30], [20, 20, 15, 15]]),
            labels=np.asarray([1, 2]),
            scores=np.asarray([0.9, 0.8])
        )
    )
    
    result = atr_inferencer.infer_from_sidd(mock_sidd_reader, image_index=0)
    
    assert isinstance(result, InferenceResult)
    assert isinstance(result.bboxes, np.ndarray)
    assert isinstance(result.labels, np.ndarray)
    assert isinstance(result.scores, np.ndarray)
    assert len(result.bboxes) == 2
    assert len(result.labels) == 2
    assert len(result.scores) == 2
    mock_inference.assert_called_once()


def test_atr_inferencer_from_file(
    atr_inferencer: ATRInferencer,
    taika_tests_path: Path,
    mocker: MockerFixture
) -> None:
    """Tests performing inference from a file path"""
    # Create a dummy file
    test_file = taika_tests_path / "test_image.ntif"
    test_file.write_text("dummy")
    
    # Mock the reader creation
    mock_reader = mocker.MagicMock(SICDReader)
    mocker.patch('taika.ml.torch_pipeline.atr_inferencer.SICDReader', return_value=mock_reader)
    
    # Mock the internal inference method
    mock_inference = mocker.patch.object(
        atr_inferencer,
        'infer_from_sicd',
        return_value=InferenceResult(
            bboxes=np.asarray([[100, 110, 25, 30]]),
            labels=np.asarray([1]),
            scores=np.asarray([0.95])
        )
    )
    
    result = atr_inferencer.infer_from_file(str(test_file))
    
    assert isinstance(result, InferenceResult)
    assert len(result.bboxes) == 1
    mock_inference.assert_called_once_with(mock_reader)


def test_atr_inferencer_with_resample(
    inference_config: InferenceConfig,
    dummy_model: ModelInterface,
    preprocessing_config: KVHolder,
    sicd_metadata: SICDType,
    mock_sicd_reader: SICDReader,
    mocker: MockerFixture
) -> None:
    """Tests ATRInferencer with resampling enabled"""
    # Update preprocessing config to include resampling
    preprocessing_config.processing_steps.resample.enabled = True
    preprocessing_config.processing_steps.resample.params.target_sample_spacing = {
        "row": 0.12,
        "col": 0.12
    }
    
    inferencer = ATRInferencer(
        model=dummy_model,
        preprocessing_config=preprocessing_config,
        inference_config=inference_config
    )
    
    # Mock the dataset creation with resampling
    mock_dataset = mocker.MagicMock(SingleImageDataset)
    mock_dataset.image_preprocessor._resampler = ResampleSAR((0.12, 0.12))
    mock_dataset.__len__ = mocker.MagicMock(return_value=5)
    
    mocker.patch.object(
        inferencer,
        '_create_dataset',
        return_value=mock_dataset
    )
    
    # Mock the inference output
    mocker.patch(
        'taika.ml.torch_pipeline.atr_inferencer.atr_image_inference',
        return_value=(
            np.asarray([[1, 52, 10, 14], [549, 574, 5, 7]]),
            np.asarray([1, 2]),
            np.asarray([0.9, 0.85])
        )
    )
    
    result = inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(result, InferenceResult)
    assert len(result.bboxes) == 2
    assert np.all(result.bboxes == np.asarray([[1, 52, 10, 14], [549, 574, 5, 7]]))


def test_atr_inferencer_class_specific_thresholds(
    dummy_model: ModelInterface,
    preprocessing_config: KVHolder,
    mock_sicd_reader: SICDReader,
    mocker: MockerFixture
) -> None:
    """Tests ATRInferencer with class-specific score thresholds"""
    # Create config with class-specific thresholds
    inference_config = InferenceConfig(
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        score_threshold={1: 0.7, 2: 0.8, 3: 0.6},
        perform_nms=True,
        nms_threshold=0.5
    )
    
    inferencer = ATRInferencer(
        model=dummy_model,
        preprocessing_config=preprocessing_config,
        inference_config=inference_config
    )
    
    # Mock the inference to return detections with varied scores
    mocker.patch(
        'taika.ml.torch_pipeline.atr_inferencer.atr_image_inference',
        return_value=(
            np.asarray([[10, 10, 20, 20], [30, 30, 20, 20], [50, 50, 20, 20]]),
            np.asarray([1, 2, 3]),
            np.asarray([0.75, 0.85, 0.65])  # Only passes respective thresholds
        )
    )
    
    result = inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(result, InferenceResult)
    assert len(result.bboxes) == 3
    assert np.all(result.scores >= 0.6)


def test_atr_inferencer_no_detections(
    atr_inferencer: ATRInferencer,
    mock_sicd_reader: SICDReader,
    mocker: MockerFixture
) -> None:
    """Tests ATRInferencer when no detections are found"""
    # Mock inference to return empty arrays
    mocker.patch(
        'taika.ml.torch_pipeline.atr_inferencer.atr_image_inference',
        return_value=(
            np.asarray([]),
            np.asarray([]),
            np.asarray([])
        )
    )
    
    result = atr_inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(result, InferenceResult)
    assert len(result.bboxes) == 0
    assert len(result.labels) == 0
    assert len(result.scores) == 0


def test_atr_inferencer_batch_processing(
    atr_inferencer: ATRInferencer,
    mock_sicd_reader: SICDReader,
    mocker: MockerFixture
) -> None:
    """Tests that ATRInferencer correctly handles batch processing"""
    # Create expected batch outputs
    expected_bboxes = np.asarray([
        [100, 110, 25, 30],
        [20, 20, 15, 15],
        [521, 21, 16, 16],
        [1022, 22, 17, 17]
    ])
    expected_labels = np.asarray([1, 2, 3, 4])
    expected_scores = np.asarray([0.9, 0.85, 0.92, 0.88])
    
    # Mock the inference
    mocker.patch(
        'taika.ml.torch_pipeline.atr_inferencer.atr_image_inference',
        return_value=(expected_bboxes, expected_labels, expected_scores)
    )
    
    result = atr_inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(result, InferenceResult)
    assert np.array_equal(result.bboxes, expected_bboxes)
    assert np.array_equal(result.labels, expected_labels)
    assert np.array_equal(result.scores, expected_scores)


def test_inference_result_filtering(
    atr_inferencer: ATRInferencer,
    mock_sicd_reader: SICDReader,
    mocker: MockerFixture
) -> None:
    """Tests that InferenceResult properly filters detections"""
    # Mock inference to return detections with mixed scores
    mocker.patch(
        'taika.ml.torch_pipeline.atr_inferencer.atr_image_inference',
        return_value=(
            np.asarray([[10, 10, 20, 20], [30, 30, 20, 20], [50, 50, 20, 20]]),
            np.asarray([1, 1, 2]),
            np.asarray([0.6, 0.4, 0.7])  # Middle detection below threshold
        )
    )
    
    # Update threshold to filter out middle detection
    atr_inferencer.inference_config.score_threshold = 0.5
    
    result = atr_inferencer.infer_from_sicd(mock_sicd_reader)
    
    # After internal filtering in atr_image_inference, should have 2 detections
    assert len(result.bboxes) == 2
    assert np.all(result.scores >= 0.5)


def test_atr_inferencer_invalid_input(
    atr_inferencer: ATRInferencer
) -> None:
    """Tests that ATRInferencer handles invalid inputs properly"""
    with pytest.raises(TypeError):
        atr_inferencer.infer_from_sicd(None)
    
    with pytest.raises(FileNotFoundError):
        atr_inferencer.infer_from_file("non_existent_file.ntif")
    
    with pytest.raises(ValueError):
        # Invalid image index for SIDD
        atr_inferencer.infer_from_sidd(MagicMock(), image_index=-1)

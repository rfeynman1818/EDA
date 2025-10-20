# ---------------------- FIXTURES ----------------------

@pytest.fixture
def inference_config() -> KVHolder:
    """Creates an inference configuration"""
    return KVHolder({
        "batch_size": 2,
        "num_workers": 1,
        "pin_memory": True,
        "score_threshold": 0.5,
        "perform_nms": True,
        "nms_threshold": 0.5,
        "augmentations": [MinMaxNorm(return_0_on_static_input=True)],
    })


@pytest.fixture
def dummy_model() -> ModelInterface:
    """A dummy model that includes a `batch_infer` method."""
    
    class DummyDataAdapter():
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
def dataset(inference_preprocessor: ImagePreprocessorAndChipper) -> SingleImageDataset:
    """A SingleImageDataset instance for inference"""
    augmentations = [MinMaxNorm(return_0_on_static_input=True)]
    return SingleImageDataset(inference_preprocessor, augmentations=augmentations)


@pytest.fixture
def atr_inferencer(
    inference_config: KVHolder,
    dummy_model: ModelInterface,
    preprocessing_config: KVHolder
) -> ATRInferencer:
    """Creates an ATR Inferencer instance"""
    return ATRInferencer(
        model=dummy_model,
        preprocessing_config=preprocessing_config,
        inference_config=inference_config
    )


# ---------------------- TESTS ----------------------

def test_atr_inferencer_init(atr_inferencer: ATRInferencer) -> None:
    """Tests ATRInferencer initialization"""
    assert atr_inferencer is not None
    assert atr_inferencer.model is not None
    assert atr_inferencer.preprocessing_config is not None
    assert atr_inferencer.inference_config is not None
    assert isinstance(atr_inferencer.inference_config, KVHolder)
    return


def test_atr_inferencer_create_dataset(
    atr_inferencer: ATRInferencer,
    mock_sicd_reader
) -> None:
    """Tests creating a dataset from the inferencer"""
    dataset = atr_inferencer._create_dataset(mock_sicd_reader)
    assert isinstance(dataset, SingleImageDataset)
    assert dataset.image_preprocessor is not None
    return


def test_atr_inferencer_from_sicd(
    atr_inferencer: ATRInferencer,
    mock_sicd_reader
) -> None:
    """Tests performing inference from a SICD reader"""
    bboxes, labels, scores = atr_inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(scores, np.ndarray)
    
    # Based on dummy model behavior with default dataset
    expected_detection_count = 2 * 9  # 2 detections per chip, 9 chips expected
    assert len(bboxes) == expected_detection_count
    assert len(labels) == expected_detection_count
    assert len(scores) == expected_detection_count
    assert np.all(scores == 1.0)
    return


def test_atr_inferencer_from_sidd(
    atr_inferencer: ATRInferencer,
    mock_sidd_reader
) -> None:
    """Tests performing inference from a SIDD reader"""
    bboxes, labels, scores = atr_inferencer.infer_from_sidd(mock_sidd_reader, image_index=0)
    
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(scores, np.ndarray)
    
    # Based on dummy model behavior
    expected_detection_count = 2 * 9  # 2 detections per chip, 9 chips expected
    assert len(bboxes) == expected_detection_count
    assert len(labels) == expected_detection_count
    assert len(scores) == expected_detection_count
    return


def test_atr_inferencer_with_resample(
    dummy_model: ModelInterface,
    preprocessing_config: KVHolder,
    mock_sicd_reader,
    sicd_metadata: SICDType
) -> None:
    """Tests ATRInferencer with resampling enabled"""
    # Update preprocessing config to include resampling
    preprocessing_config.processing_steps.resample.enabled = True
    preprocessing_config.processing_steps.resample.params.target_sample_spacing = {
        "row": 0.12,
        "col": 0.12
    }
    
    inference_config = KVHolder({
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": True,
        "score_threshold": 0.5,
        "perform_nms": True,
        "nms_threshold": 0.5,
        "augmentations": [MinMaxNorm(return_0_on_static_input=True)],
    })
    
    inferencer = ATRInferencer(
        model=dummy_model,
        preprocessing_config=preprocessing_config,
        inference_config=inference_config
    )
    
    bboxes, labels, scores = inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert len(bboxes) > 0
    assert len(labels) > 0
    assert len(scores) > 0
    return


def test_atr_inferencer_class_specific_thresholds(
    dummy_model: ModelInterface,
    preprocessing_config: KVHolder,
    mock_sicd_reader
) -> None:
    """Tests ATRInferencer with class-specific score thresholds"""
    
    class VariedScoreModel(ModelInterface):
        """Model that returns varied scores for testing threshold filtering"""
        def __init__(self):
            self.bboxes = np.asarray([[10, 10, 20, 20], [30, 30, 20, 20], [50, 50, 20, 20]])
            self.labels = np.asarray([1, 2, 3])
            self.scores = np.asarray([0.75, 0.85, 0.65])
            self._data_adapter = DataAdapter()
        
        def batch_infer(self, chip_ids: list[int]) -> tuple:
            """Returns detections with varied scores"""
            n_chips = len(chip_ids)
            all_bboxes = []
            all_labels = []
            all_scores = []
            all_chip_ids = []
            
            for chip_id in chip_ids:
                all_bboxes.append(self.bboxes)
                all_labels.append(self.labels)
                all_scores.append(self.scores)
                all_chip_ids.extend([chip_id] * len(self.bboxes))
            
            return (
                np.concatenate(all_bboxes) if all_bboxes else np.array([]),
                np.array(all_chip_ids),
                np.concatenate(all_labels) if all_labels else np.array([]),
                np.concatenate(all_scores) if all_scores else np.array([])
            )
        
        @property
        def data_adapter(self) -> DataAdapter:
            class Adapter():
                @staticmethod
                def to_model(input: Any, target: dict) -> np.ndarray:
                    return np.asarray([x[ATRSampleKeys.CHIP_INDEX] for x in target])
                
                @staticmethod
                def from_model(model_outputs: tuple) -> tuple:
                    bboxes, chip_ids, labels, scores = model_outputs
                    return chip_ids, bboxes, labels, scores
            
            return Adapter()
    
    # Create config with class-specific thresholds
    inference_config = KVHolder({
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": True,
        "score_threshold": {1: 0.7, 2: 0.8, 3: 0.6},
        "perform_nms": False,
        "nms_threshold": 0.5
    })
    
    inferencer = ATRInferencer(
        model=VariedScoreModel(),
        preprocessing_config=preprocessing_config,
        inference_config=inference_config
    )
    
    bboxes, labels, scores = inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(scores, np.ndarray)
    # All detections should pass their respective thresholds
    assert len(bboxes) > 0
    assert len(labels) > 0
    assert len(scores) > 0
    return


def test_atr_inferencer_no_detections(
    preprocessing_config: KVHolder,
    mock_sicd_reader
) -> None:
    """Tests ATRInferencer when no detections are found"""
    
    class NoDetectionModel(ModelInterface):
        """Model that returns no detections"""
        def batch_infer(self, chip_ids: list[int]) -> tuple:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        @property
        def data_adapter(self) -> DataAdapter:
            class Adapter():
                @staticmethod
                def to_model(input: Any, target: dict) -> np.ndarray:
                    return np.asarray([x[ATRSampleKeys.CHIP_INDEX] for x in target])
                
                @staticmethod
                def from_model(model_outputs: tuple) -> tuple:
                    bboxes, chip_ids, labels, scores = model_outputs
                    return chip_ids, bboxes, labels, scores
            
            return Adapter()
    
    inference_config = KVHolder({
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": True,
        "score_threshold": 0.5,
        "perform_nms": True,
        "nms_threshold": 0.5
    })
    
    inferencer = ATRInferencer(
        model=NoDetectionModel(),
        preprocessing_config=preprocessing_config,
        inference_config=inference_config
    )
    
    bboxes, labels, scores = inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert len(bboxes) == 0
    assert len(labels) == 0
    assert len(scores) == 0
    return


def test_atr_inferencer_batch_processing(
    atr_inferencer: ATRInferencer,
    mock_sicd_reader
) -> None:
    """Tests that ATRInferencer correctly handles batch processing"""
    # Update config to use different batch size
    atr_inferencer.inference_config.batch_size = 3
    
    bboxes, labels, scores = atr_inferencer.infer_from_sicd(mock_sicd_reader)
    
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(scores, np.ndarray)
    
    # Verify we get expected number of detections regardless of batch size
    expected_detection_count = 2 * 9  # 2 detections per chip, 9 chips
    assert len(bboxes) == expected_detection_count
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
    return

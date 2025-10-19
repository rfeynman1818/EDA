# -------------------- TEST IMPLEMENTATIONS --------------------
class TestDataAdapter(DataAdapter):
    """DataAdapter for testing"""
    
    @staticmethod
    def to_model(input: Any, target: dict) -> np.ndarray:
        """Converts from sdk inputs to model inputs"""
        return np.asarray([x[ATRSampleKeys.CHIP_INDEX] for x in target])
    
    @staticmethod
    def from_model(model_outputs: tuple) -> tuple:
        """Converts from model outputs to sdk outputs"""
        bboxes, chip_ids, labels, scores = model_outputs
        return chip_ids, bboxes, labels, scores


class TestModel(ModelInterface):
    """Test implementation of ModelInterface"""
    
    def __init__(self):
        """Initialization"""
        self.bboxes = np.asarray([[10, 20, 30, 40]])
        self.labels = np.asarray([1])
        self._data_adapter = TestDataAdapter()
        self.weights_loaded = False
    
    def load_weights(self, weights_path: str) -> None:
        """Mock weight loading"""
        self.weights_loaded = True
    
    def batch_infer(self, batch_inputs: Any) -> tuple:
        """Creates dummy detections"""
        return (
            self.bboxes,
            np.asarray([0]),  # chip_ids
            self.labels,
            np.asarray([0.9])  # scores
        )
    
    @property
    def data_adapter(self) -> DataAdapter:
        """Returns the model's data adapter"""
        return self._data_adapter
    
    @classmethod
    def load_from_config(cls, config: KVHolder) -> 'TestModel':
        """Load from config"""
        return cls()


class TestPreprocessorWizard:
    """Minimal implementation of preprocessor wizard for testing"""
    
    def create_preprocessor(self, image: Any, metadata: Any) -> 'TestPreprocessor':
        """Create a test preprocessor"""
        return TestPreprocessor()
    
    @classmethod
    def from_preprocessor_config(cls, config: Any, for_inference: bool = True) -> 'TestPreprocessorWizard':
        """Create from config"""
        return cls()


class TestPreprocessor:
    """Minimal preprocessor for testing"""
    
    def __init__(self):
        self.resampler = None
        self.image_sample_spacing = None


# -------------------- FIXTURES --------------------
@pytest.fixture
def model() -> TestModel:
    """Test model instance"""
    return TestModel()


@pytest.fixture
def preprocessor_wizard() -> TestPreprocessorWizard:
    """Test preprocessor wizard"""
    return TestPreprocessorWizard()


@pytest.fixture
def class_map() -> dict[int, str]:
    """Sample class mapping"""
    return {0: "background", 1: "vehicle", 2: "building"}


@pytest.fixture
def test_config() -> KVHolder:
    """Create a test configuration"""
    # This would need actual KVHolder implementation
    # For now, creating a minimal mock structure
    class MockKVHolder:
        def __init__(self, data=None):
            self._data = data or {}
        
        def __getattr__(self, name):
            if name in self._data:
                if isinstance(self._data[name], dict):
                    return MockKVHolder(self._data[name])
                return self._data[name]
            return MockKVHolder()
        
        def to_dict(self):
            return self._data
    
    config_data = {
        'data': {
            'preprocessing': {}
        },
        'class_map': {0: "bg", 1: "obj"},
        'inference': {
            'environment': {
                'test_env': {
                    'batch_size': 8,
                    'num_workers': 4
                }
            },
            'dataloader': {
                'transforms': {}
            },
            'nms': {
                'enabled': True,
                'nms_threshold': 0.7
            },
            'score_threshold': 0.6
        }
    }
    
    return MockKVHolder(config_data)


@pytest.fixture
def atr_inferencer(model: TestModel, preprocessor_wizard: TestPreprocessorWizard, 
                   class_map: dict[int, str]) -> 'ATRInferencer':
    """ATRInferencer instance for testing"""
    from atr_inferencer import ATRInferencer
    
    return ATRInferencer(
        model=model,
        preprocessor_wizard=preprocessor_wizard,
        augmentations=None,
        class_map=class_map,
        batch_size=2,
        num_workers=0,
        score_threshold=0.5,
        perform_nms=True,
        nms_threshold=0.5
    )


# -------------------- TESTS --------------------
def test_init(model: TestModel, preprocessor_wizard: TestPreprocessorWizard, 
              class_map: dict[int, str]) -> None:
    """Test ATRInferencer initialization"""
    from atr_inferencer import ATRInferencer
    
    inferencer = ATRInferencer(
        model=model,
        preprocessor_wizard=preprocessor_wizard,
        augmentations=None,
        class_map=class_map,
        batch_size=4,
        num_workers=2,
        score_threshold=0.3,
        perform_nms=False,
        nms_threshold=None
    )
    
    assert inferencer._model == model
    assert inferencer._preprocessor_wizard == preprocessor_wizard
    assert inferencer._augmentations is None
    assert inferencer._class_map == class_map
    assert inferencer._batch_size == 4
    assert inferencer._num_workers == 2
    assert inferencer._score_threshold == 0.3
    assert inferencer._perform_nms == False
    assert inferencer._nms_threshold is None
    assert isinstance(inferencer._algorithm_metadata, dict)
    assert len(inferencer._algorithm_metadata) == 0


def test_infer_invalid_path_type(atr_inferencer: 'ATRInferencer') -> None:
    """Test infer with invalid path type"""
    with pytest.raises(TypeError, match="image_path should be of type"):
        atr_inferencer.infer(123)
    
    with pytest.raises(TypeError, match="image_path should be of type"):
        atr_inferencer.infer(None)


def test_infer_file_not_found(atr_inferencer: 'ATRInferencer') -> None:
    """Test infer with non-existent file"""
    with pytest.raises(FileNotFoundError, match="File cannot be found"):
        atr_inferencer.infer("non_existent_file.ntf")
    
    with pytest.raises(FileNotFoundError, match="File cannot be found"):
        atr_inferencer.infer(Path("non_existent_file.sicd"))


def test_call_method(atr_inferencer: 'ATRInferencer') -> None:
    """Test __call__ method as alias for infer"""
    # Create a temporary file to avoid FileNotFoundError
    with tempfile.NamedTemporaryFile(suffix='.ntf') as tmp:
        tmp_path = tmp.name
        
        # This will fail at read_sicd, but we can verify __call__ invokes infer
        try:
            result1 = atr_inferencer.infer(tmp_path)
        except Exception as e1:
            pass
        
        try:
            result2 = atr_inferencer(tmp_path)
        except Exception as e2:
            pass
        
        # Both should fail at the same point (read_sicd)
        assert type(e1) == type(e2)


def test_algorithm_metadata_property(atr_inferencer: 'ATRInferencer') -> None:
    """Test algorithm_metadata property getter and setter"""
    # Initial state
    assert atr_inferencer.algorithm_metadata == {}
    
    # Set metadata
    test_metadata = {
        "algorithm_name": "ATR_Model_v2",
        "algorithm_version": "2.0.1",
        "training_date": "2025-01-01"
    }
    atr_inferencer.algorithm_metadata = test_metadata
    assert atr_inferencer.algorithm_metadata == test_metadata
    assert atr_inferencer._algorithm_metadata == test_metadata


def test_boxes_to_polygon_empty_array(atr_inferencer: 'ATRInferencer') -> None:
    """Test _boxes_to_polygon with empty array
    Note: This is one of the few methods we can test without full dependencies
    """
    # Create minimal metadata structure
    class MinimalMetadata:
        class ImageData:
            NumRows = 1000
            NumCols = 1200
    
    metadata = MinimalMetadata()
    bboxes = np.array([])
    
    # This should handle empty arrays gracefully
    polygons = atr_inferencer._boxes_to_polygon(bboxes, metadata)
    assert polygons.size == 0


# -------------------- LIMITED INTEGRATION TESTS --------------------
def test_load_from_config_structure() -> None:
    """Test that load_from_config has correct structure
    Note: Cannot fully test without all dependencies
    """
    from atr_inferencer import ATRInferencer
    
    # Verify the method exists and has correct signature
    assert hasattr(ATRInferencer, 'load_from_config')
    assert callable(ATRInferencer.load_from_config)
    
    # Check it's a classmethod
    import inspect
    assert isinstance(inspect.getattr_static(ATRInferencer, 'load_from_config'), classmethod)


def test_model_interface_compliance(model: TestModel) -> None:
    """Verify our test model complies with ModelInterface"""
    # Check required methods exist
    assert hasattr(model, 'load_weights')
    assert hasattr(model, 'batch_infer')
    assert hasattr(model, 'data_adapter')
    assert hasattr(model, 'load_from_config')
    
    # Check data_adapter has required methods
    adapter = model.data_adapter
    assert hasattr(adapter, 'to_model')
    assert hasattr(adapter, 'from_model')
    
    # Test basic functionality
    model.load_weights("dummy_path.pth")
    assert model.weights_loaded == True
    
    outputs = model.batch_infer(np.array([0]))
    assert len(outputs) == 4  # bboxes, chip_ids, labels, scores

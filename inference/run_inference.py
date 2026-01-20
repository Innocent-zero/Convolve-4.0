from utils.extractors import VLMExtractor, OCRExtractor, CVExtractor
from inference.ensemble_v2 import EnsembleExtractorV2
from utils.preprocessing import DocumentPreprocessor
from models.spatial_graph_attention import SpatialGraphAttention
from inference.sgan_extractor import SGANExtractor

# Load SGAN
sgan = SGANExtractor(
    checkpoint_path="checkpoints/iteration_3/best_model.pt"
)

# Load fallback extractors
vlm = VLMExtractor()
ocr = OCRExtractor()
cv = CVExtractor()

# Create ensemble
ensemble = EnsembleExtractorV2(
    sgan=sgan,
    vlm=vlm,
    ocr=ocr,
    cv=cv
)

# Run inference
preprocessor = DocumentPreprocessor()
doc = "data/invoices/invoice_042.pdf"
processed = preprocessor.process(doc)

results = ensemble.extract_standard(
    images=processed["images"],
    language=processed["language"]
)

print(results)

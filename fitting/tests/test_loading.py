import logging
import sys
from pathlib import Path

from fitting2.core.formatting import formatFromMetadata
from fitting2.data.loading import FileLoader, extractHistogram, histToBinnedData, hasVariationAxis, variationNames
from fitting2.pipeline import PipelineConfig, runPipeline

logging.basicConfig(level=logging.INFO)

def test_loading_directly():
    print("--- Test Direct Loading ---")
    bkg_path = Path("testexport/2018/Signal312/qcd_inclusive_2018/comp_mStop.pklz4")
    sig_path = Path("testexport/2018/Signal312/signal_2018_312_1000_400_official/comp_mStop.pklz4")
    
    loader = FileLoader.forPath(bkg_path)
    print(f"Bkg loader type: {type(loader).__name__}")
    
    bkg_raw = loader.load(bkg_path)
    bkg_hist = extractHistogram(bkg_raw)
    
    print(f"Bkg hist has variation axis: {hasVariationAxis(bkg_hist)}")
    print(f"Bkg variations: {variationNames(bkg_hist)}")
    
    bkg_data = histToBinnedData(bkg_hist, variation="central")
    print(f"Bkg central BinnedData shape: X={bkg_data.X.shape}, Y={bkg_data.Y.shape}")
    
    # Signal
    sig_raw = FileLoader.forPath(sig_path).load(sig_path)
    sig_hist = extractHistogram(sig_raw)
    
    print()
    print(f"Signal variations count: {len(variationNames(sig_hist))}")
    print(f"First 5 sig variations: {variationNames(sig_hist)[:5]}")
    
    sig_data = histToBinnedData(sig_hist, variation="central")
    print(f"Sig central BinnedData shape: X={sig_data.X.shape}, Y={sig_data.Y.shape}")
    
    # Metadata string formatting
    if "metadata" in sig_raw:
        meta = sig_raw["metadata"]
        try:
            formatted = formatFromMetadata(
                "Era: {era.name}, Lumi: {era.lumi}, "
                "mStop: {other_data.stop_mass}, mChi: {other_data.chargino_mass}",
                meta
            )
            print()
            print(f"Formatted metadata: {formatted}")
        except Exception as e:
            print(f"Error formatting metadata: {e}")

def test_pipeline():
    print("\n--- Test Pipeline Initiation ---")
    config = PipelineConfig(
        background_path=Path(sys.argv[1] if len(sys.argv) > 1 else "testexport/2018/Signal312/qcd_inclusive_2018/comp_mStop.pklz4"),
        signal_path=Path(sys.argv[2] if len(sys.argv) > 2 else "testexport/2018/Signal312/signal_2018_312_1000_400_official/comp_mStop.pklz4"),
        output_dir=Path("test_output"),
        optimization=None # Keep fast for test
    )
    # Just configure it minimally to pass preprocessing/normalization
    # or just run it and let it fail if optimization is not fully setup
    # actually we can just pass a dummy optimization config to fail fast or let it run
    
if __name__ == "__main__":
    test_loading_directly()

#!/bin/bash
# Quick start script for fine-tuning FLAN-T5 model
# Usage: bash training/run_training.sh

set -e

echo "================================"
echo "🚀 FLAN-T5 Fine-Tuning Pipeline"
echo "================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check dependencies
echo -e "${YELLOW}[1/4]${NC} Checking dependencies..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || exit 1
python -c "import transformers; print(f'  ✓ Transformers {transformers.__version__}')" || exit 1
python -c "import datasets; print(f'  ✓ Datasets')" || exit 1

DEVICE=$(python -c "import torch; print('GPU (CUDA)' if torch.cuda.is_available() else 'CPU')")
echo -e "  ✓ Device detected: ${GREEN}${DEVICE}${NC}"

# Step 2: Prepare data
echo -e "\n${YELLOW}[2/4]${NC} Preparing training data..."
python training/prepare_data.py
echo -e "  ${GREEN}✓ Data ready!${NC}"

# Step 3: Fine-tune
echo -e "\n${YELLOW}[3/4]${NC} Starting fine-tuning..."
echo "  This may take 1-2 hours on CPU, or 15-30 min on GPU..."
python training/finetune.py
echo -e "  ${GREEN}✓ Fine-tuning complete!${NC}"

# Step 4: Test inference
echo -e "\n${YELLOW}[4/4]${NC} Testing inference..."
python training/inference.py
echo -e "  ${GREEN}✓ Inference test passed!${NC}"

# Summary
echo ""
echo -e "${GREEN}================================"
echo "✅ All steps complete!"
echo "================================${NC}"
echo ""
echo "Next steps:"
echo "1. Update .env with:"
echo "   FINETUNED_MODEL_PATH=training/finetuned_model"
echo ""
echo "2. Restart the app:"
echo "   python run.py"
echo ""
echo "3. The chatbot will now use your fine-tuned model! 🎉"

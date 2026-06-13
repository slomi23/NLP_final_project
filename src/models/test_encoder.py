# test_encoder_backprop.py
import torch
import torch.nn as nn
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Added to Python path: {project_root}")

# Now the import should work
from models.encoder import EncoderOnlyTransformer  # Replace with actual encoder name

def test_encoder_backprop():
    """Test if your encoder supports backpropagation"""
    print("🚀 Testing Encoder Backpropagation Support...")
    print("=" * 50)
    
    try:
        # Import your encoder
        from models.encoder import EncoderOnlyTransformer  # Replace with actual name
        
        # Create encoder
        encoder = EncoderOnlyTransformer()
        encoder.train()  # Important!
        
        print("✅ Encoder imported successfully")
        print(f"✅ Encoder in training mode: {encoder.training}")
        
        # Check if parameters require gradients
        requires_grad_count = 0
        total_params = 0
        
        for name, param in encoder.named_parameters():
            total_params += 1
            if param.requires_grad:
                requires_grad_count += 1
                print(f"✅ {name}: requires_grad=True")
            else:
                print(f"❌ {name}: requires_grad=False")
        
        print(f"\n📊 Gradient Support Summary:")
        print(f"   Total parameters: {total_params}")
        print(f"   Parameters with gradients: {requires_grad_count}")
        print(f"   Gradient support: {requires_grad_count/total_params*100:.1f}%")
        
        # Test forward pass
        dummy_input = torch.randn(4, 384)  # Adjust size as needed
        with torch.no_grad():
            output = encoder(dummy_input)
        
        print(f"✅ Forward pass works: output shape {output.shape}")
        
        # Test backward pass
        encoder.train()  # Make sure it's in training mode
        dummy_input.requires_grad_(True)  # Make input require gradients
        output = encoder(dummy_input)
        
        # Create a dummy loss
        loss = output.mean()
        
        # Backward pass
        try:
            loss.backward()
            print("✅ Backward pass completed successfully")
            
            # Check gradients
            has_gradients = False
            for param in encoder.parameters():
                if param.grad is not None and param.grad.norm() > 0:
                    has_gradients = True
                    break
            
            if has_gradients:
                print("✅ Encoder gradients computed successfully")
            else:
                print("❌ No encoder gradients found")
                
        except Exception as e:
            print(f"❌ Backward pass failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import encoder: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing encoder: {e}")
        return False

if __name__ == "__main__":
    success = test_encoder_backprop()
    if success:
        print("\n🎉 Encoder supports backpropagation!")
    else:
        print("\n⚠️  Encoder needs fixes for backpropagation.")

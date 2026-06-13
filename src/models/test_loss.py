# test_loss.py (simple version)
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import sys
import os

# Add your project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss import (
    ContrastiveLoss, 
    NTXentLoss, 
    InfoNCELoss, 
    SupervisedContrastiveLoss, 
    LiftedStructureLoss,
    create_loss
)

def test_contrastive_loss():
    """Test ContrastiveLoss"""
    print("🔍 Testing ContrastiveLoss...")
    
    try:
        loss_fn = ContrastiveLoss(margin=1.0, reduction='mean')
        
        # Create test data
        torch.manual_seed(42)
        emb1 = torch.randn(4, 384, requires_grad=True)
        emb2 = torch.randn(4, 384, requires_grad=True)
        labels = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        
        # Test similar pairs
        sim_loss = loss_fn(emb1[:2], emb2[:2], labels[:2])
        print(f"  Similar pairs loss: {sim_loss.item():.4f}")
        
        # Test dissimilar pairs
        dissim_loss = loss_fn(emb1[2:], emb2[2:], labels[2:])
        print(f"  Dissimilar pairs loss: {dissim_loss.item():.4f}")
        
        # Test gradients
        sim_loss.backward()
        if emb1.grad is not None and emb1.grad.norm() > 0:
            print("  ✅ Gradients work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_nt_xent_loss():
    """Test NTXentLoss"""
    print("🔍 Testing NTXentLoss...")
    
    try:
        loss_fn = NTXentLoss(temperature=0.07, reduction='mean')
        
        # Create test data
        torch.manual_seed(42)
        batch_size = 8
        embeddings = torch.randn(2 * batch_size, 384, requires_grad=True)
        
        # Test loss
        loss = loss_fn(embeddings)
        print(f"  NT-Xent loss: {loss.item():.4f}")
        
        # Test gradients
        loss.backward()
        if embeddings.grad is not None and embeddings.grad.norm() > 0:
            print("  ✅ Gradients work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_info_nce_loss():
    """Test InfoNCELoss"""
    print("🔍 Testing InfoNCELoss...")
    
    try:
        loss_fn = InfoNCELoss(temperature=0.07, reduction='mean')
        
        # Create test data
        torch.manual_seed(42)
        batch_size = 6
        anchor = torch.randn(batch_size, 384, requires_grad=True)
        positive = torch.randn(batch_size, 384, requires_grad=True)
        negatives = torch.randn(batch_size, 1, 384, requires_grad=True)  # Only 1 negative for simplicity
        
        # Test loss
        loss = loss_fn(anchor, positive, negatives)
        print(f"  InfoNCE loss: {loss.item():.4f}")
        
        # Test gradients
        loss.backward()
        if anchor.grad is not None and anchor.grad.norm() > 0:
            print("  ✅ Gradients work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_supervised_contrastive_loss():
    """Test SupervisedContrastiveLoss"""
    print("🔍 Testing SupervisedContrastiveLoss...")
    
    try:
        loss_fn = SupervisedContrastiveLoss(temperature=0.07, reduction='mean')
        
        # Create test data
        torch.manual_seed(42)
        batch_size = 16
        embeddings = torch.randn(batch_size, 384, requires_grad=True)
        labels = torch.randint(0, 4, (batch_size,))
        
        # Test loss
        loss = loss_fn(embeddings, labels)
        print(f"  Supervised Contrastive loss: {loss.item():.4f}")
        
        # Test gradients
        loss.backward()
        if embeddings.grad is not None and embeddings.grad.norm() > 0:
            print("  ✅ Gradients work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_lifted_structure_loss():
    """Test LiftedStructureLoss"""
    print("🔍 Testing LiftedStructureLoss...")
    
    try:
        loss_fn = LiftedStructureLoss(margin=1.0, reduction='mean')
        
        # Create test data
        torch.manual_seed(42)
        batch_size = 16
        embeddings = torch.randn(batch_size, 384, requires_grad=True)
        labels = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
        
        # Test loss
        loss = loss_fn(embeddings, labels)
        print(f"  Lifted Structure loss: {loss.item():.4f}")
        
        # Test gradients
        loss.backward()
        if embeddings.grad is not None and embeddings.grad.norm() > 0:
            print("  ✅ Gradients work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_factory_function():
    """Test factory function"""
    print("🔍 Testing factory function...")
    
    try:
        # Test creating different losses
        losses_to_test = ['contrastive', 'nt_xent', 'info_nce', 'supervised_contrastive', 'lifted_structure']
        
        for loss_name in losses_to_test:
            if loss_name in ['contrastive', 'lifted_structure']:
                loss_fn = create_loss(loss_name, margin=1.0)
            else:
                loss_fn = create_loss(loss_name, temperature=0.07)
            print(f"  ✅ Successfully created {loss_name} loss")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Simple Loss Function Tests...")
    print("=" * 50)
    
    tests = [
        test_contrastive_loss,
        test_nt_xent_loss,
        test_info_nce_loss,
        test_supervised_contrastive_loss,
        test_lifted_structure_loss,
        test_factory_function
    ]
    
    results = {}
    for test in tests:
        results[test.__name__] = test()
        print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if passed_test:
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed.")
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
    print("\n🏁 Testing Complete!")

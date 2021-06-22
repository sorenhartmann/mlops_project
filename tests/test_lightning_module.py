import torch


def test_forward(model, small_batch):

    batch_size = small_batch[0].shape[0]

    output = model.forward(
        input_ids=small_batch[0],
        token_type_ids=small_batch[1],
        attention_mask=small_batch[2],
        labels=small_batch[3].unsqueeze(1),
    )

    assert output.logits.shape == (batch_size, 2)
    assert not torch.isnan(output.loss)
    assert not torch.isinf(output.loss)
    assert output.loss.requires_grad


def test_training_step(model, small_batch):

    batch_idx = list(range(small_batch[0].shape[0]))

    loss = model.training_step(small_batch, batch_idx)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.requires_grad

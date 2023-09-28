import torch

def format_number(num, length=20):
    num_str = str(num)
    if "." in num_str:
        integer_part, decimal_part = num_str.split(".")
        formatted_decimal_part = decimal_part.ljust(3, '0')
        formatted_integer_part = integer_part
        formatted_num = "{}.{}".format(formatted_integer_part, formatted_decimal_part)
    else:
        formatted_num = num_str
    formatted_num = formatted_num.ljust(length, '0')
    return formatted_num

def save_checkpoint(file_path, model, optimizer, loss, epoch):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)
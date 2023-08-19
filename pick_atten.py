import torch
import pickle
from nltk.corpus import stopwords


class AttentionValue():
    def __init__(self, 
                num_input, 
                num_words, 
                scale, 
                dir,
                percentage,
                largest=True):
        # scale can either be actual number of scale you want to attack
        # or the percentage of the ranked pixels that picked for attack.
        self.num_input = num_input
        self.num_words = num_words
        self.largest = largest
        self.scale = scale
        self.dir = dir
        self.percentage = percentage


    def find_largest_top_k(self, 
                        tensor,
                        scale
                        ):
        """
        Return the top-K items and its index of a given tensor.
        """
        # Flatten the tensor
        flattened = tensor.flatten()

        # If select according to percentage.
        if self.percentage:
            scale = int(scale * len(flattened))
        # Use torch.topk() to find the largest top-k elements
        top_k_values, top_k_indices_flattened = torch.topk(flattened, scale, largest=self.largest)
        
        # Reshape the flattened indices to match the original tensor shape
        original_shape = tensor.shape
        top_k_indices = [
            (int(index // original_shape[1]), int(index % original_shape[1]))
            for index in top_k_indices_flattened
        ]
        
        return top_k_values.numpy(), top_k_indices

    def pick_pixels_with_weight(self, 
                                weight, 
                                num_samples, 
                                image_size=255):

        # Flatten the image tensor and weight array
        weight_flat = weight.view(-1)

        # Normalize weights to add up to 1
        normalized_weights = weight_flat / torch.sum(weight_flat)

        # Generate random indices based on weights
        # indices = torch.arange(image_flat.size(0))
        selected_indices = torch.multinomial(normalized_weights, num_samples=num_samples, replacement=True)

        width = image_size
        selected_x = (selected_indices % width).tolist()
        selected_y = (selected_indices // width).tolist()
        selected_coordinates = list(zip(selected_x, selected_y))
        selected_weights = weight_flat[selected_indices]

        return selected_weights.numpy(), selected_coordinates
        
    def __get_item__(self, 
                    filter=True, 
                    condition='all-in', 
                    pick_method='topk'):
        """
        Get image pixel indexes for attack.

        Inputs:
                scale -> numbers of scale to attack. <Change to percentage if needed>, must be integer 
                         if in 'pick_pixels_with_weight' case.
                filter -> whether to filter out keywords,
                condition -> 'all-in' or 'separate',
                pick_method -> 'topk' or 'withweight',
                largest -> topk (forward or reverse order)
        Outputs:
                The focused pixels index of the input sample (on all the keywords.)
        """
        examples = self.num_input
        with open(self.dir, 'rb') as file:
            data = pickle.load(file)
        """
        data -> [id, [alpha_resized, words]] * num_input
        """
        stopword = set(stopwords.words('english'))
        focused = []
        for i in range(self.num_input):
            focus = []
            attens = data[i][1][0]
            if filter:
                filtered_words = [(index, word) for index, word in enumerate(data[i][1][1]) if word.lower() not in stopword]
                if len(filtered_words) < 1:
                    examples -= 1
                    continue
                if len(filtered_words) < self.num_words:
                    self.num_words = len(filtered_words)
                words = filtered_words[:self.num_words]
            else: 
                words = data[i][1][1] # Not filtering any words.
                words = [(index, word) for index, word in enumerate(data[i][1][1])]

            if condition == "separate":
                # In this case, we search top k (scale) for each word from its attention.
                # Should return focused -> number = 'pixels' for every word * num_input.
                for words_ in words:
                    focus.extend(self.find_largest_top_k(attens[words_[0]], scale=self.scale)[1])
                focused.append(focus)
                # Deal duplicate
                # focused = list(set(focused))
            if condition == 'all-in':
                # In this case, our search space is top k 'pixels' across ALL words.
                overall = torch.zeros_like(data[0][1][0][0])
                for words_ in words:
                    overall = torch.add(overall, data[i][1][0][words_[0]])
                if pick_method == 'topk':
                    focused.append(self.find_largest_top_k(overall, scale=self.scale)[1])
                if pick_method == 'withweight':
                    focused.append(self.pick_pixels_with_weight(overall, num_samples=self.scale)[1])
        return focused

def out_attention(focused):
    # get pixels that not in <focused>, <focused> is a list with num_input size.
    base = torch.zeros((255, 255))
    out_atts = []
    out_att = [(row, col) for row in range(base.size(0)) for col in range(base.size(1)) if (row, col) not in focused]
    return out_att

if __name__ == '__main__':
    
    wu = AttentionValue(10, 50, 0.5, 'attens_10samples.json', percentage=True)

    att = wu.__get_item__(False, 'separate', pick_method='topk')[4]

    print(len(att))

from django.shortcuts import render
from transformers import AutoModelForCausalLM , AutoTokenizer
from django.views.decorators.csrf import csrf_exempt
import torch
import json
from django.http import JsonResponse
import os
import difflib


def index(request):
    return render(request,'index.html')

#Lad tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
# there three type of pretrain models
# 1 microsoft/DialoGPT-large  50GB + Model will be downloed
# 2 microsoft/DialoGPT-medium 25GB +
# 3 microsoft/DialoGPT-samll  450+ mb files will be downloaded n

# processors and gpu

specific_responses_file = os.path.join(os.path.dirname(__file__), 'faq.json')
with open(specific_responses_file, 'r') as f:
    specific_responses_data = json.load(f)


specific_responses = {
    item['input']:item['response'] for item in specific_responses_data
}

def get_closest_match(user_message,possible_inputs,threshold=0.6):
    close_match = difflib.get_close_matches(user_message,possible_inputs,n=1,cutoff=threshold)
    return close_match[0] if close_match else None

# fuzzymatchin
# waat is webcog : "It's a Youtube Channel

# Setup device for model (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message')

        possible_inputs=list(specific_responses.keys())
        closest_match=get_closest_match(user_message,possible_inputs)

        if closest_match:
            response=specific_responses[closest_match]
            context = {
                'response':response,
                'correct_question':closest_match
            }
            return JsonResponse(context)
        else:

            # tokienize userinput and genrete output response
            inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
            attention_mask = torch.ones(inputs.shape, device=device)
            outputs = model.generate(inputs,attention_mask=attention_mask,max_length=1000,
                                     pad_token_id=tokenizer.eos_token_id)

            response = tokenizer.decode(outputs[:,inputs.shape[-1]:][0], skip_special_tokens=True)

            return JsonResponse({'response':response})
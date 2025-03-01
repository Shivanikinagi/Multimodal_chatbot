from Task1.Untitled import generate_summary
from Task2.chatbot import multimodal_chatbot

def main():
    print("Welcome to the Multi-Modal Chatbot and Text Summarization Tool!")
    print("Choose an option:")
    print("1. Summarize Text")
    print("2. Chat with the Multi-Modal Chatbot")
    print("3. Exit")

    while True:
        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            text = input("Enter the text to summarize: ")
            summary = generate_summary(text, summary_length=3)
            print(f"Summary:\n{summary}")

        elif choice == "2":
            multimodal_chatbot()

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
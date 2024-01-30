from openai import OpenAI
import time

client = OpenAI()

ASSIST_SUM_ID = "asst_DwZv7cRrD8cnVh5MSnSix7HX"

# get full regulation given the paragrah


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


def create_thread_and_run(user_input, assist_id):
    thread = client.beta.threads.create()
    run = submit_message(assist_id, thread, user_input)
    return thread, run


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def summarize(quote):
    thread, run = create_thread_and_run(quote, ASSIST_SUM_ID)
    run = wait_on_run(run, thread)

    return get_response(thread).data[-1].content[0].text.value

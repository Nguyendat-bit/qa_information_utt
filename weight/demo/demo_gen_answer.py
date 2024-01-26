import sys 
import json 
sys.path.append('../')

from call_api_process import call_generate_answer
import gradio as gr 


context= """+ Sứ mạng:
“Trường Đại học Công nghệ Giao thông vận tải có sứ mạng đào tạo và cung cấp nguồn nhân lực chất lượng cao theo hướng ứng dụng, đa ngành, đa lĩnh vực, nghiên cứu khoa học và chuyển giao công nghệ phục vụ sự nghiệp phát triển của ngành Giao thông vận tải và của đất nước, phù hợp với xu thế phát triển quốc tế, hội nhập với nền giáo dục đại học tiên tiến của khu vực và trên thế giới”.
+Tầm nhìn:
Đến năm 2030, có một số ngành đào tạo ngang tầm với các trường đại học có uy tín trong khu vực và trên thế giới; là trung tâm nghiên cứu khoa học ứng dụng, chuyển giao công nghệ mới và hợp tác quốc tế trong lĩnh vực Giao thông vận tải.
Đến năm 2045, chào mừng kỷ niệm 100 năm thành lập Trường, trở thành trường đại học thông minh, trung tâm nghiên cứu khoa học, chuyển giao công nghệ và hợp tác quốc tế trong lĩnh vực Giao thông vận tải
+ Giá trị cốt lõi:
Đoàn kết - Trí tuệ - Đổi mới - Hội nhập - Phát triển bền vững
Đoàn kết: Tập thể sư phạm Nhà trường là một khối thống nhất, đồng tâm nhất trí vì sự phát triển của Nhà trường; luôn sẵn sàng hợp tác, chia sẻ mọi nguồn lực trong công việc, hỗ trợ và giúp đỡ lẫn nhau để hoàn thành tốt mọi nhiệm vụ; lợi ích của mỗi cá nhân trong Trường gắn liền với sự phát triển của Nhà trường. Cựu sinh viên, học viên, sinh viên và các đối tác luôn là một phần gắn bó chặt chẽ của Trường Đại học Công nghệ GTVT.
Trí tuệ và Đổi mới: Trường Đại học Công nghệ GTVT đề cao trí tuệ và đổi mới sáng tạo, coi trí tuệ là tài sản và dùng đổi mới sáng tạo để: Tối ưu hóa – Đơn giản hóa – Khác biệt hóa; xây dựng môi trường học tập và nghiên cứu thân thiện, đảm bảo và tạo điều kiện tối đa cán bộ, giảng viên, sinh viên, học viên được tự do đổi mới sáng tạo, phát triển tư duy.
Hội nhập: Trường Đại học Công nghệ GTVT đẩy mạnh hợp tác với các trường đại học, các tổ chức và cá nhân trong và ngoài nước nhằm tạo điều kiện tối đa cho cán bộ giảng viên, người lao động và học viên, sinh viên có cơ hội tiếp cận và hội nhập với tiêu chuẩn của nền giáo dục đại học tiên tiến trong khu vực và trên thế giới.
Phát triển bền vững: Các hoạt động của Trường Đại học Công nghệ GTVT luôn hướng tới sự phát triển trên nguyên tắc bảo vệ môi trường, phục vụ cộng đồng, bảo đảm công bằng xã hội, tôn trọng các quyền con người, bảo đảm sự bình đẳng giữa các thế hệ.
+ Triết lý giáo dục: Ứng dụng- Thực học- Thực nghiệp
Ứng dụng: Các chương trình đào tạo được Nhà trường xây theo định hướng ứng dụng có mục tiêu và nội dung theo hướng phát triển kết quả nghiên cứu cơ bản, ứng dụng các công nghệ nguồn thành các giải pháp công nghệ, quy trình quản lý, thiết kế các công cụ hoàn chỉnh phục vụ nhu cầu của thực tiễn sản xuất.
Thực học: Các chương trình đào tạo được xây dựng đảm bảo tỷ lệ thực hành, thực tập trong trường và ngoài doanh nghiệp chiếm từ 40% trở lên; được tổ chức dạy thật, học thật, thi thật.
Thực nghiệp: Các chương trình đào tạo được xây dựng gắn liền với nhu cầu của doanh nghiệp trong và ngoài nước đảm bảo sinh viên được tuyển dụng ngay sau khi tốt nghiệp ra trường."""

question=  "bạn hãy trình bày tầm nhìn , giá trị cốt lỗi, triết lý giáo dục của trường"



def predict(context, question, temperature, top_p, top_k, new_tokens, type_prompt):
    return call_generate_answer(question, context, temperature= temperature, 
                                top_k= top_k, top_p= top_p, new_tokens= new_tokens, type_prompt= type_prompt)

gr.Interface(
    fn= predict,
    inputs=[
        gr.Textbox(lines=7, value=context, label="Context Paragraph"),
        gr.Textbox(lines=2, value=question, label="Question"),

        gr.components.Slider(minimum=0, maximum=1, value=0.56, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.9, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=100, label="Top k"),
        gr.components.Slider(
            minimum=1, maximum=512, step=1, value=500, label="Max tokens"
        ),
        gr.Dropdown(
            ['short', 'creative', 'introduce', 'precise'], label= 'Type Prompt'
        )
    ],
    title="QA Generation",
    outputs=[gr.Textbox(label="Answer")],
).launch()



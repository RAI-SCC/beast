class TestImportModelFactory:
    def test_import_model_factory(self) -> None:
        from beast import get_model

        assert get_model is not None
        assert callable(get_model)


# class TestModelFactory:
#     def test_get_model_with_str_id(self) -> None:
#         assert False

#     def test_get_model_with_config_dict(self) -> None:
#         assert False

#     def test_get_model_with_keyword_args(self) -> None:
#         assert False

#     def test_get_model_with_invalid_input(self) -> None:
#         assert False

#     def test_get_model_with_missing_required_args(self) -> None:
#         assert False

#     def test_get_model_with_extra_args(self) -> None:
#         assert False

#     def test_get_model_with_conflicting_args(self) -> None:
#         assert False

#     def test_get_model_with_id_and_config(self) -> None:
#         assert False

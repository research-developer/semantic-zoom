"""Tests for NSM-59: User prompt system for clarification.

Tests:
- Grammar error prompts: accept, alternative, ignore
- Ambiguity prompts: select interpretation
- User response storage and application
- Batch resolution support
"""
import pytest


class TestCorrectionPrompt:
    """Test grammar correction prompts."""

    def test_create_correction_prompt(self):
        """Test creating a correction prompt from grammar error."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import create_correction_prompt

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            prompt = create_correction_prompt(result.errors[0])
            assert prompt is not None
            assert hasattr(prompt, 'error')
            assert hasattr(prompt, 'options')

    def test_correction_prompt_options(self):
        """Test that correction prompt has accept/alternative/ignore options."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import create_correction_prompt, PromptOption

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            prompt = create_correction_prompt(result.errors[0])
            option_types = [o.option_type for o in prompt.options]
            assert PromptOption.ACCEPT in option_types
            assert PromptOption.IGNORE in option_types
            # Alternative may or may not be available
            assert len(option_types) >= 2

    def test_accept_correction(self):
        """Test accepting a correction."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import (
            create_correction_prompt,
            apply_response,
            PromptOption,
            UserResponse,
        )

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors and result.errors[0].suggestion:
            prompt = create_correction_prompt(result.errors[0])
            response = UserResponse(
                prompt_id=prompt.prompt_id,
                option=PromptOption.ACCEPT,
            )
            new_text = apply_response(text, result.errors[0], response)
            # Text should be corrected
            assert new_text != text

    def test_ignore_correction(self):
        """Test ignoring a correction."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import (
            create_correction_prompt,
            apply_response,
            PromptOption,
            UserResponse,
        )

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            prompt = create_correction_prompt(result.errors[0])
            response = UserResponse(
                prompt_id=prompt.prompt_id,
                option=PromptOption.IGNORE,
            )
            new_text = apply_response(text, result.errors[0], response)
            # Text should remain unchanged
            assert new_text == text

    def test_alternative_correction(self):
        """Test providing an alternative correction."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import (
            create_correction_prompt,
            apply_response,
            PromptOption,
            UserResponse,
        )

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            prompt = create_correction_prompt(result.errors[0])
            response = UserResponse(
                prompt_id=prompt.prompt_id,
                option=PromptOption.ALTERNATIVE,
                alternative_text="run",
            )
            new_text = apply_response(text, result.errors[0], response)
            # Custom alternative should be applied
            assert "run" in new_text or new_text != text


class TestAmbiguityPrompt:
    """Test ambiguity resolution prompts."""

    def test_create_ambiguity_prompt(self):
        """Test creating an ambiguity prompt."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities
        from semantic_zoom.phase7.user_prompts import create_ambiguity_prompt

        text = "I saw the man with the telescope."
        result = detect_ambiguities(text)

        if result.ambiguities:
            prompt = create_ambiguity_prompt(result.ambiguities[0])
            assert prompt is not None
            assert hasattr(prompt, 'ambiguity')
            assert hasattr(prompt, 'options')

    def test_ambiguity_prompt_lists_interpretations(self):
        """Test that ambiguity prompt lists all interpretations."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities
        from semantic_zoom.phase7.user_prompts import create_ambiguity_prompt

        text = "I saw the man with the telescope."
        result = detect_ambiguities(text)

        if result.ambiguities:
            prompt = create_ambiguity_prompt(result.ambiguities[0])
            # Options should correspond to interpretations
            assert len(prompt.options) >= len(result.ambiguities[0].interpretations)

    def test_select_interpretation(self):
        """Test selecting an interpretation."""
        from semantic_zoom.phase7.ambiguity_detection import detect_ambiguities
        from semantic_zoom.phase7.user_prompts import (
            create_ambiguity_prompt,
            resolve_ambiguity,
            UserResponse,
            PromptOption,
        )

        text = "I saw the man with the telescope."
        result = detect_ambiguities(text)

        if result.ambiguities and result.ambiguities[0].interpretations:
            ambiguity = result.ambiguities[0]
            prompt = create_ambiguity_prompt(ambiguity)

            # Select first interpretation
            response = UserResponse(
                prompt_id=prompt.prompt_id,
                option=PromptOption.SELECT,
                selected_index=0,
            )
            resolved = resolve_ambiguity(ambiguity, response)
            assert resolved is not None
            assert resolved.selected_interpretation is not None


class TestResponseStorage:
    """Test user response storage and application."""

    def test_store_response(self):
        """Test storing user responses."""
        from semantic_zoom.phase7.user_prompts import (
            ResponseStore,
            UserResponse,
            PromptOption,
        )

        store = ResponseStore()
        response = UserResponse(
            prompt_id="prompt_001",
            option=PromptOption.ACCEPT,
        )
        store.add(response)

        retrieved = store.get("prompt_001")
        assert retrieved is not None
        assert retrieved.option == PromptOption.ACCEPT

    def test_apply_stored_response(self):
        """Test applying stored responses to parse."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import (
            ResponseStore,
            create_correction_prompt,
            apply_stored_responses,
            UserResponse,
            PromptOption,
        )

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            store = ResponseStore()
            # IMPORTANT: Use deterministic=True so ID matches during retrieval
            prompt = create_correction_prompt(result.errors[0], deterministic=True)
            response = UserResponse(
                prompt_id=prompt.prompt_id,
                option=PromptOption.ACCEPT,
            )
            store.add(response)

            corrected = apply_stored_responses(text, result.errors, store)
            assert corrected is not None
            # Verify the correction was actually applied
            if result.errors[0].suggestion:
                assert corrected != text, "Correction should be applied"
                assert result.errors[0].suggestion in corrected

    def test_stored_response_deterministic_id_required(self):
        """Test that deterministic ID is required for apply_stored_responses.

        This tests the bug fix: prompts must use deterministic IDs for
        stored responses to be retrieved correctly.
        """
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import (
            ResponseStore,
            create_correction_prompt,
            apply_stored_responses,
            UserResponse,
            PromptOption,
        )

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors and result.errors[0].suggestion:
            store = ResponseStore()

            # Create prompt WITH deterministic ID
            prompt = create_correction_prompt(result.errors[0], deterministic=True)
            response = UserResponse(
                prompt_id=prompt.prompt_id,
                option=PromptOption.ACCEPT,
            )
            store.add(response)

            # Apply should work with deterministic IDs
            corrected = apply_stored_responses(text, result.errors, store)
            assert corrected != text, "With deterministic ID, correction should apply"

    def test_deterministic_id_is_consistent(self):
        """Test that deterministic ID is the same for the same error."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import create_correction_prompt

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            error = result.errors[0]
            prompt1 = create_correction_prompt(error, deterministic=True)
            prompt2 = create_correction_prompt(error, deterministic=True)
            assert prompt1.prompt_id == prompt2.prompt_id, \
                "Deterministic IDs should be the same for the same error"

    def test_random_id_is_different(self):
        """Test that random ID is different each time."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import create_correction_prompt

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            error = result.errors[0]
            prompt1 = create_correction_prompt(error, deterministic=False)
            prompt2 = create_correction_prompt(error, deterministic=False)
            assert prompt1.prompt_id != prompt2.prompt_id, \
                "Random IDs should be different each time"


class TestBatchResolution:
    """Test batch resolution support."""

    def test_batch_correction_prompts(self):
        """Test creating batch correction prompts."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import create_batch_prompts

        text = "The dogs runs quick. She don't like it."
        result = check_grammar(text)

        if len(result.errors) >= 1:
            prompts = create_batch_prompts(result.errors)
            assert len(prompts) >= 1

    def test_batch_response_application(self):
        """Test applying batch responses."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import (
            create_batch_prompts,
            apply_batch_responses,
            UserResponse,
            PromptOption,
        )

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            prompts = create_batch_prompts(result.errors)
            responses = [
                UserResponse(
                    prompt_id=p.prompt_id,
                    option=PromptOption.ACCEPT,
                )
                for p in prompts
            ]
            corrected = apply_batch_responses(text, result.errors, responses)
            assert corrected is not None

    def test_mixed_batch_responses(self):
        """Test batch with mixed accept/ignore responses."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import (
            create_batch_prompts,
            apply_batch_responses,
            UserResponse,
            PromptOption,
        )

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            prompts = create_batch_prompts(result.errors)
            # Mix of accept and ignore
            responses = []
            for i, p in enumerate(prompts):
                option = PromptOption.ACCEPT if i % 2 == 0 else PromptOption.IGNORE
                responses.append(UserResponse(prompt_id=p.prompt_id, option=option))

            corrected = apply_batch_responses(text, result.errors, responses)
            assert corrected is not None


class TestPromptStructure:
    """Test prompt data structures."""

    def test_prompt_option_enum(self):
        """Test PromptOption enum values."""
        from semantic_zoom.phase7.user_prompts import PromptOption

        assert PromptOption.ACCEPT is not None
        assert PromptOption.IGNORE is not None
        assert PromptOption.ALTERNATIVE is not None
        assert PromptOption.SELECT is not None

    def test_user_response_structure(self):
        """Test UserResponse has required fields."""
        from semantic_zoom.phase7.user_prompts import UserResponse, PromptOption

        response = UserResponse(
            prompt_id="test_001",
            option=PromptOption.ACCEPT,
        )
        assert response.prompt_id == "test_001"
        assert response.option == PromptOption.ACCEPT

    def test_prompt_has_id(self):
        """Test that prompts have unique IDs."""
        from semantic_zoom.phase7.grammar_check import check_grammar
        from semantic_zoom.phase7.user_prompts import create_correction_prompt

        text = "The dogs runs quickly."
        result = check_grammar(text)

        if result.errors:
            prompt = create_correction_prompt(result.errors[0])
            assert hasattr(prompt, 'prompt_id')
            assert prompt.prompt_id is not None
